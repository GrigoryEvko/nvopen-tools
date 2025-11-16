// Function: sub_2E46380
// Address: 0x2e46380
//
__int64 __fastcall sub_2E46380(__int64 a1, unsigned int a2, unsigned int a3, __int64 *a4, __int64 a5)
{
  __int64 result; // rax
  __int64 *v7; // r12
  __int16 v8; // dx
  __int64 v9; // rbx
  unsigned int v10; // esi
  __int64 v11; // rcx
  __int64 v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // r15
  unsigned int v15; // esi
  unsigned int v17; // [rsp+4h] [rbp-3Ch]
  __int64 v18; // [rsp+8h] [rbp-38h]

  result = (__int64)&a4[a5];
  v18 = result;
  if ( a4 != (__int64 *)result )
  {
    v7 = a4;
    v17 = a2 - 1;
    while ( 1 )
    {
      result = *v7;
      v8 = *(_WORD *)(*v7 + 68);
      if ( v8 == 14 )
        break;
      if ( v8 == 15 )
      {
        v11 = *(_QWORD *)(result + 32);
        result = 5LL * (*(_DWORD *)(result + 40) & 0xFFFFFF);
        v12 = v11 + 8 * result;
        v13 = v11 + 80;
LABEL_13:
        if ( v12 == v13 )
        {
LABEL_10:
          if ( (__int64 *)v18 == ++v7 )
            return result;
        }
        else
        {
          v14 = v13;
          do
          {
            if ( !*(_BYTE *)v14 )
            {
              result = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)a1 + 16LL));
              v15 = *(_DWORD *)(v14 + 8);
              if ( a2 == v15
                || v15 - 1 <= 0x3FFFFFFE && v17 <= 0x3FFFFFFE && (result = sub_E92070(result, v15, a2), (_BYTE)result) )
              {
                result = sub_2EAB0C0(v14, a3);
              }
            }
            v14 += 40;
          }
          while ( v12 != v14 );
          if ( (__int64 *)v18 == ++v7 )
            return result;
        }
      }
      else
      {
        if ( v8 != 17 )
          BUG();
        v9 = *(_QWORD *)(result + 32);
        if ( *(_BYTE *)v9 )
          goto LABEL_10;
        result = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)a1 + 16LL));
        v10 = *(_DWORD *)(v9 + 8);
        if ( a2 != v10 )
        {
          if ( v10 - 1 > 0x3FFFFFFE )
            goto LABEL_10;
          if ( v17 > 0x3FFFFFFE )
            goto LABEL_10;
          result = sub_E92070(result, v10, a2);
          if ( !(_BYTE)result )
            goto LABEL_10;
        }
        ++v7;
        result = sub_2EAB0C0(v9, a3);
        if ( (__int64 *)v18 == v7 )
          return result;
      }
    }
    v13 = *(_QWORD *)(result + 32);
    v12 = v13 + 40;
    goto LABEL_13;
  }
  return result;
}
