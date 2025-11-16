// Function: sub_EB9B50
// Address: 0xeb9b50
//
__int64 __fastcall sub_EB9B50(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned __int8 v4; // al
  __int64 v5; // rbx
  int v6; // eax
  int v7; // eax
  __int64 v8; // rbx
  int v9; // eax
  int v10; // eax
  __int64 v11; // rdx
  size_t **v12; // rax
  size_t *v13; // r14
  _QWORD *v14; // rbx
  _QWORD *v15; // r12
  _QWORD *v16; // r15
  __int64 v17; // rbx
  __int64 v18; // r12
  __int64 v19; // rdi
  const void *v20; // [rsp+0h] [rbp-C0h]
  const void *v21; // [rsp+0h] [rbp-C0h]
  __int64 v22; // [rsp+0h] [rbp-C0h]
  size_t v23; // [rsp+8h] [rbp-B8h]
  size_t v24; // [rsp+8h] [rbp-B8h]
  _QWORD *v25; // [rsp+8h] [rbp-B8h]
  __int64 v26; // [rsp+18h] [rbp-A8h] BYREF
  const void *v27; // [rsp+20h] [rbp-A0h] BYREF
  size_t v28; // [rsp+28h] [rbp-98h]
  _QWORD v29[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v30; // [rsp+50h] [rbp-70h]
  _QWORD v31[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v32; // [rsp+80h] [rbp-40h]

  v27 = 0;
  v28 = 0;
  v26 = 0;
  if ( (unsigned __int8)sub_ECD7C0(a1, &v26) )
    return 1;
  v31[0] = "expected identifier in '.purgem' directive";
  v32 = 259;
  v4 = sub_EB61F0(a1, (__int64 *)&v27);
  if ( (unsigned __int8)sub_ECE070(a1, v4, v26, v31) )
    return 1;
  v2 = sub_ECE000(a1);
  if ( (_BYTE)v2 )
  {
    return 1;
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 224);
    v20 = v27;
    v23 = v28;
    v6 = sub_C92610();
    v7 = sub_C92860((__int64 *)(v5 + 2384), v20, v23, v6);
    if ( v7 == -1 || 8LL * *(unsigned int *)(v5 + 2392) == 8LL * v7 )
    {
      v30 = 1283;
      v29[0] = "macro '";
      v32 = 770;
      v29[2] = v27;
      v29[3] = v28;
      v31[0] = v29;
      v31[2] = "' is not defined";
      return (unsigned int)sub_ECDA70(a1, a2, v31, 0, 0);
    }
    else
    {
      v8 = *(_QWORD *)(a1 + 224);
      v21 = v27;
      v24 = v28;
      v9 = sub_C92610();
      v10 = sub_C92860((__int64 *)(v8 + 2384), v21, v24, v9);
      if ( v10 != -1 )
      {
        v11 = *(_QWORD *)(v8 + 2384);
        v12 = (size_t **)(v11 + 8LL * v10);
        if ( v12 != (size_t **)(v11 + 8LL * *(unsigned int *)(v8 + 2392)) )
        {
          v13 = *v12;
          sub_C929B0(v8 + 2384, *v12);
          v14 = (_QWORD *)v13[9];
          v15 = (_QWORD *)v13[8];
          v22 = *v13 + 97;
          if ( v14 != v15 )
          {
            do
            {
              if ( (_QWORD *)*v15 != v15 + 2 )
                j_j___libc_free_0(*v15, v15[2] + 1LL);
              v15 += 4;
            }
            while ( v14 != v15 );
            v15 = (_QWORD *)v13[8];
          }
          if ( v15 )
            j_j___libc_free_0(v15, v13[10] - (_QWORD)v15);
          v16 = (_QWORD *)v13[5];
          v25 = (_QWORD *)v13[6];
          if ( v25 != v16 )
          {
            do
            {
              v17 = v16[3];
              v18 = v16[2];
              if ( v17 != v18 )
              {
                do
                {
                  if ( *(_DWORD *)(v18 + 32) > 0x40u )
                  {
                    v19 = *(_QWORD *)(v18 + 24);
                    if ( v19 )
                      j_j___libc_free_0_0(v19);
                  }
                  v18 += 40;
                }
                while ( v17 != v18 );
                v18 = v16[2];
              }
              if ( v18 )
                j_j___libc_free_0(v18, v16[4] - v18);
              v16 += 6;
            }
            while ( v25 != v16 );
            v16 = (_QWORD *)v13[5];
          }
          if ( v16 )
            j_j___libc_free_0(v16, v13[7] - (_QWORD)v16);
          sub_C7D6A0((__int64)v13, v22, 8);
        }
      }
    }
  }
  return v2;
}
