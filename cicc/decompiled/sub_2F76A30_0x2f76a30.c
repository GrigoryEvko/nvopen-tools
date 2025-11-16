// Function: sub_2F76A30
// Address: 0x2f76a30
//
__int64 __fastcall sub_2F76A30(
        unsigned __int16 *a1,
        char a2,
        _QWORD *a3,
        char a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        __int128 a8)
{
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned int *v12; // rbx
  __int64 result; // rax
  int v14; // edi
  __int64 v15; // rax
  unsigned __int16 *v16; // r8
  unsigned __int16 v17; // dx
  unsigned int v18; // esi
  unsigned int v19; // ecx
  unsigned __int16 v20; // si
  unsigned __int16 *v21; // rcx
  unsigned __int16 i; // r10
  unsigned __int16 v23; // r11
  int v24; // edx
  __int64 j; // rdx
  unsigned __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+10h] [rbp-40h]

  v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a3 + 16LL) + 200LL))(*(_QWORD *)(*a3 + 16LL));
  v11 = v10;
  if ( (a7 & 0x80000000) != 0 )
  {
    v27 = v10;
    v26 = *(_QWORD *)(a3[7] + 16LL * (a7 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
    v12 = (unsigned int *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 416LL))(v10);
    result = *(unsigned int *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v27 + 376LL))(v27, v26);
  }
  else
  {
    v12 = (unsigned int *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v10 + 424LL))(v10, a7);
    result = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v11 + 384LL))(v11, a7);
  }
  if ( *v12 == -1 )
    v12 = 0;
  v14 = -(int)result;
  if ( !a2 )
    v14 = result;
  if ( !a4 )
    goto LABEL_12;
  result = *((_QWORD *)&a8 + 1) | a8;
  if ( a8 != 0 )
  {
    v15 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a3 + 16LL) + 200LL))(*(_QWORD *)(*a3 + 16LL));
    v14 = (*(__int64 (__fastcall **)(__int64, _QWORD *, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v15 + 440LL))(
            v15,
            a3,
            a7,
            a8,
            *((_QWORD *)&a8 + 1));
    result = (unsigned int)-v14;
    if ( a2 )
      v14 = -v14;
LABEL_12:
    v16 = a1 + 32;
    do
    {
      if ( !v12 )
        return result;
      result = (__int64)a1;
      v17 = *a1;
      if ( *a1 )
      {
        while ( 1 )
        {
          v18 = *v12;
          v19 = v17 - 1;
          if ( *v12 <= v19 )
            break;
          result += 4;
          if ( (unsigned __int16 *)result == v16 )
            return result;
          v17 = *(_WORD *)result;
          if ( !*(_WORD *)result )
            goto LABEL_18;
        }
        if ( v16 == (unsigned __int16 *)result )
          return result;
        if ( v18 == v19 )
        {
          v24 = v14 + *(__int16 *)(result + 2);
          if ( !v24 )
            goto LABEL_32;
          goto LABEL_26;
        }
      }
      else
      {
LABEL_18:
        if ( v16 == (unsigned __int16 *)result )
          return result;
        v18 = *v12;
      }
      v20 = v18 + 1;
      if ( v20 )
      {
        v21 = (unsigned __int16 *)result;
        for ( i = 0; ; i = v23 )
        {
          v23 = v21[1];
          *v21 = v20;
          v21 += 2;
          *(v21 - 1) = i;
          if ( !v17 || v16 == v21 )
            break;
          v20 = v17;
          v17 = *v21;
        }
      }
      v24 = v14 + *(__int16 *)(result + 2);
      if ( !v24 )
      {
LABEL_32:
        for ( j = result + 4; v16 != (unsigned __int16 *)j; j += 4 )
        {
          if ( !*(_WORD *)(result + 4) )
            break;
          *(_DWORD *)result = *(_DWORD *)(result + 4);
          result = j;
        }
        *(_DWORD *)result = 0;
        goto LABEL_27;
      }
LABEL_26:
      *(_WORD *)(result + 2) = v24;
LABEL_27:
      ++v12;
    }
    while ( *v12 != -1 );
  }
  return result;
}
