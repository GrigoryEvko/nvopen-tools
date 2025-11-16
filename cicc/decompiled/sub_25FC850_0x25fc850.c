// Function: sub_25FC850
// Address: 0x25fc850
//
__int64 __fastcall sub_25FC850(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  unsigned int *v5; // rdx
  __int64 v6; // r15
  unsigned int *v7; // r12
  unsigned int *v8; // rbx
  __int64 v9; // rdx
  __int64 *v10; // r8
  __int64 v11; // rcx
  __int64 v12; // rax
  unsigned int v13; // [rsp-5Ch] [rbp-5Ch]
  __int64 *v14; // [rsp-58h] [rbp-58h]
  __int64 v15; // [rsp-50h] [rbp-50h]
  __int64 v16; // [rsp-40h] [rbp-40h] BYREF

  result = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)result )
  {
    v5 = *(unsigned int **)(a1 + 176);
    v6 = *(_QWORD *)(a1 + 296);
    v7 = &v5[4 * *(unsigned int *)(a1 + 192)];
    if ( v5 != v7 )
    {
      while ( 1 )
      {
        result = *v5;
        v8 = v5;
        if ( (unsigned int)result <= 0xFFFFFFFD )
          break;
        v5 += 4;
        if ( v7 == v5 )
          return result;
      }
      if ( v7 != v5 )
      {
        v9 = *(_QWORD *)(v6 + 56);
        v10 = (__int64 *)*((_QWORD *)v8 + 1);
        if ( (*(_BYTE *)(v9 + 2) & 1) != 0 )
          goto LABEL_14;
        while ( 1 )
        {
          v11 = 5 * result;
          v12 = *(_QWORD *)(v9 + 96);
          v16 = v9;
          v8 += 4;
          a2 = (__int64 *)(v12 + 8 * v11);
          result = sub_BD79D0(v10, a2, (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_25F6110, (__int64)&v16);
          if ( v8 == v7 )
            break;
          while ( *v8 > 0xFFFFFFFD )
          {
            v8 += 4;
            if ( v7 == v8 )
              return result;
          }
          if ( v8 == v7 )
            return result;
          v9 = *(_QWORD *)(v6 + 56);
          result = *v8;
          v10 = (__int64 *)*((_QWORD *)v8 + 1);
          if ( (*(_BYTE *)(v9 + 2) & 1) != 0 )
          {
LABEL_14:
            v13 = result;
            v14 = v10;
            v15 = v9;
            sub_B2C6D0(v9, (__int64)a2, v9, a4);
            result = v13;
            v10 = v14;
            v9 = v15;
          }
        }
      }
    }
  }
  return result;
}
