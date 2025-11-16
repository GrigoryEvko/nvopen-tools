// Function: sub_D6E1B0
// Address: 0xd6e1b0
//
__int64 __fastcall sub_D6E1B0(__int64 *a1, __int64 a2, __int64 a3, __int64 **a4, __int64 a5, char a6)
{
  __int64 result; // rax
  __int64 v9; // r12
  __int64 **v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 *v15; // rdx
  int v16; // edx
  unsigned int v17; // r14d
  __int64 v18; // rsi
  __int64 v19; // r13
  __int64 v20; // r15
  char *v21; // rax
  char *v22; // rcx
  char *v23; // rdx
  char *v24; // rax
  __int64 *v25; // rax
  __int64 v26; // [rsp+0h] [rbp-120h]
  int v27; // [rsp+10h] [rbp-110h]
  _BYTE v30[48]; // [rsp+20h] [rbp-100h] BYREF
  __int64 v31; // [rsp+50h] [rbp-D0h] BYREF
  char *v32; // [rsp+58h] [rbp-C8h]
  __int64 v33; // [rsp+60h] [rbp-C0h]
  int v34; // [rsp+68h] [rbp-B8h]
  char v35; // [rsp+6Ch] [rbp-B4h]
  char v36; // [rsp+70h] [rbp-B0h] BYREF

  result = sub_D68B40(*a1, a2);
  if ( result )
  {
    v9 = result;
    if ( (unsigned __int8)sub_AA5590(a2, 1) )
    {
      return sub_1041EA0(*a1, v9, a3, 0);
    }
    else
    {
      v10 = &a4[a5];
      v11 = sub_10420D0(*a1, a3);
      v35 = 1;
      v26 = v11;
      v31 = 0;
      v32 = &v36;
      v33 = 16;
      v34 = 0;
      while ( v10 != a4 )
      {
        v15 = *a4++;
        sub_D695C0((__int64)v30, (__int64)&v31, v15, v12, v13, v14);
      }
      v16 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
      if ( v16 )
      {
        v17 = 0;
        do
        {
          v18 = *(_QWORD *)(v9 - 8);
          v19 = *(_QWORD *)(v18 + 32LL * *(unsigned int *)(v9 + 76) + 8LL * v17);
          v20 = *(_QWORD *)(v18 + 32LL * v17);
          if ( v35 )
          {
            v21 = v32;
            v22 = &v32[8 * HIDWORD(v33)];
            if ( v32 != v22 )
            {
              while ( v19 != *(_QWORD *)v21 )
              {
                v21 += 8;
                if ( v22 == v21 )
                  goto LABEL_18;
              }
LABEL_12:
              sub_D689D0(v26, v20, v19);
              if ( !a6 )
              {
                if ( v35 )
                {
                  v23 = &v32[8 * HIDWORD(v33)];
                  if ( v32 != v23 )
                  {
                    v24 = v32;
                    while ( v19 != *(_QWORD *)v24 )
                    {
                      v24 += 8;
                      if ( v23 == v24 )
                        goto LABEL_13;
                    }
                    --HIDWORD(v33);
                    *(_QWORD *)v24 = *(_QWORD *)&v32[8 * HIDWORD(v33)];
                    ++v31;
                  }
                }
                else
                {
                  v25 = sub_C8CA60((__int64)&v31, v19);
                  if ( v25 )
                  {
                    *v25 = -2;
                    ++v34;
                    ++v31;
                  }
                }
              }
LABEL_13:
              sub_D68A80(v9, v17);
              v16 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
              continue;
            }
LABEL_18:
            ++v17;
          }
          else
          {
            v27 = v16;
            if ( sub_C8CA60((__int64)&v31, v19) )
              goto LABEL_12;
            v16 = v27;
            ++v17;
          }
        }
        while ( v17 != v16 );
      }
      sub_D689D0(v9, v26, a3);
      result = sub_D6D630((__int64)a1, v26);
      if ( !v35 )
        return _libc_free(v32, v26);
    }
  }
  return result;
}
