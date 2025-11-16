// Function: sub_1EE7080
// Address: 0x1ee7080
//
__int64 __fastcall sub_1EE7080(__int64 a1, int a2, char a3, _QWORD *a4)
{
  __int64 v5; // r14
  __int64 (*v7)(void); // rax
  __int64 v8; // rdx
  unsigned int *v9; // r12
  __int64 result; // rax
  unsigned __int16 *v11; // rdi
  int v12; // r10d
  unsigned __int16 v13; // dx
  unsigned int v14; // esi
  unsigned int v15; // ecx
  unsigned __int16 v16; // si
  unsigned __int16 *v17; // rcx
  unsigned __int16 i; // r8
  unsigned __int16 v19; // r9
  int v20; // edx
  int v21; // edx
  _DWORD *v22; // rdx
  unsigned __int64 v23; // r13
  __int64 v24; // rax
  _QWORD *v25; // [rsp+8h] [rbp-38h]

  v5 = 0;
  v7 = *(__int64 (**)(void))(**(_QWORD **)(*a4 + 16LL) + 112LL);
  if ( v7 != sub_1D00B10 )
  {
    v25 = a4;
    v24 = v7();
    a4 = v25;
    v5 = v24;
  }
  v8 = *(_QWORD *)v5;
  if ( a2 < 0 )
  {
    v23 = *(_QWORD *)(a4[3] + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
    v9 = (unsigned int *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(v8 + 224))(v5, v23);
    result = *(unsigned int *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v5 + 184LL))(v5, v23);
  }
  else
  {
    v9 = (unsigned int *)(*(__int64 (__fastcall **)(__int64, _QWORD))(v8 + 232))(v5, (unsigned int)a2);
    result = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v5 + 192LL))(v5, (unsigned int)a2);
  }
  v11 = (unsigned __int16 *)(a1 + 64);
  if ( *v9 == -1 )
    v9 = 0;
  v12 = -(int)result;
  if ( !a3 )
    v12 = result;
  while ( 2 )
  {
    if ( v9 )
    {
      result = a1;
      while ( 1 )
      {
        v13 = *(_WORD *)result;
        if ( !*(_WORD *)result )
        {
          if ( v11 == (unsigned __int16 *)result )
            return result;
          v14 = *v9;
          goto LABEL_17;
        }
        v14 = *v9;
        v15 = v13 - 1;
        if ( v15 >= *v9 )
          break;
        result += 4;
        if ( (unsigned __int16 *)result == v11 )
          return result;
      }
      if ( v11 == (unsigned __int16 *)result )
        return result;
      if ( v15 == v14 )
      {
        v20 = v12 + *(__int16 *)(result + 2);
        if ( !v20 )
          goto LABEL_29;
LABEL_23:
        *(_WORD *)(result + 2) = v20;
        goto LABEL_24;
      }
LABEL_17:
      v16 = v14 + 1;
      if ( v16 )
      {
        v17 = (unsigned __int16 *)result;
        for ( i = 0; ; i = v19 )
        {
          v19 = v17[1];
          *v17 = v16;
          v17 += 2;
          *(v17 - 1) = i;
          if ( !v13 || v11 == v17 )
            break;
          v16 = v13;
          v13 = *v17;
        }
      }
      v20 = v12 + *(__int16 *)(result + 2);
      if ( v20 )
        goto LABEL_23;
LABEL_29:
      result += 4;
      if ( v11 != (unsigned __int16 *)result )
      {
        while ( 1 )
        {
          v22 = (_DWORD *)(result - 4);
          if ( !*(_WORD *)result )
            break;
          v21 = *(_DWORD *)result;
          result += 4;
          *(_DWORD *)(result - 8) = v21;
          if ( v11 == (unsigned __int16 *)result )
            goto LABEL_24;
        }
        if ( v11 != (unsigned __int16 *)result )
        {
          result = *(unsigned int *)result;
          *v22 = result;
        }
      }
LABEL_24:
      if ( *++v9 != -1 )
        continue;
    }
    return result;
  }
}
