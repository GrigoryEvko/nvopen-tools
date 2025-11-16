// Function: sub_38F4810
// Address: 0x38f4810
//
__int64 __fastcall sub_38F4810(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned int v3; // ecx
  unsigned int v4; // r13d
  unsigned __int8 v6; // al
  int v7; // eax
  __int64 v8; // rbx
  int v9; // eax
  __int64 v10; // rdx
  size_t **v11; // rax
  __int64 v12; // rdi
  size_t *v13; // rbx
  size_t *v14; // rax
  size_t v15; // rbx
  unsigned __int64 v16; // r14
  __int64 v17; // r15
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rdi
  __int64 v20; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v21; // [rsp+8h] [rbp-98h]
  __int64 v22; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int8 *v23; // [rsp+20h] [rbp-80h] BYREF
  size_t v24; // [rsp+28h] [rbp-78h]
  _QWORD v25[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v26; // [rsp+40h] [rbp-60h]
  _QWORD v27[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v28; // [rsp+60h] [rbp-40h]

  v23 = 0;
  v24 = 0;
  v22 = 0;
  if ( (unsigned __int8)sub_3909470(a1, &v22) )
    return 1;
  v26 = 259;
  v25[0] = "expected identifier in '.purgem' directive";
  v6 = sub_38F0EE0(a1, (__int64 *)&v23, v2, v3);
  if ( (unsigned __int8)sub_3909C80(a1, v6, v22, v25) )
    return 1;
  v27[0] = "unexpected token in '.purgem' directive";
  v28 = 259;
  v4 = sub_3909E20(a1, 9, v27);
  if ( (_BYTE)v4 )
  {
    return 1;
  }
  else
  {
    v20 = *(_QWORD *)(a1 + 320);
    v7 = sub_16D1B30((__int64 *)(v20 + 1488), v23, v24);
    if ( v7 == -1 || 8LL * *(unsigned int *)(v20 + 1496) == 8LL * v7 )
    {
      v26 = 1283;
      v28 = 770;
      v25[0] = "macro '";
      v25[1] = &v23;
      v27[0] = v25;
      v27[1] = "' is not defined";
      return (unsigned int)sub_3909790(a1, a2, v27, 0, 0);
    }
    else
    {
      v8 = *(_QWORD *)(a1 + 320);
      v9 = sub_16D1B30((__int64 *)(v8 + 1488), v23, v24);
      if ( v9 != -1 )
      {
        v10 = *(_QWORD *)(v8 + 1488);
        v11 = (size_t **)(v10 + 8LL * v9);
        if ( v11 != (size_t **)(v10 + 8LL * *(unsigned int *)(v8 + 1496)) )
        {
          v12 = v8 + 1488;
          v13 = *v11;
          v21 = (unsigned __int64)*v11;
          sub_16D1CB0(v12, *v11);
          v14 = v13;
          v15 = v13[6];
          v16 = v14[5];
          if ( v15 != v16 )
          {
            do
            {
              v17 = *(_QWORD *)(v16 + 24);
              v18 = *(_QWORD *)(v16 + 16);
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
                  v18 += 40LL;
                }
                while ( v17 != v18 );
                v18 = *(_QWORD *)(v16 + 16);
              }
              if ( v18 )
                j_j___libc_free_0(v18);
              v16 += 48LL;
            }
            while ( v15 != v16 );
            v16 = *(_QWORD *)(v21 + 40);
          }
          if ( v16 )
            j_j___libc_free_0(v16);
          _libc_free(v21);
        }
      }
    }
  }
  return v4;
}
