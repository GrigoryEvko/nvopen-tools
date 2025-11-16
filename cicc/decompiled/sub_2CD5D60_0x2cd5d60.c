// Function: sub_2CD5D60
// Address: 0x2cd5d60
//
__int64 __fastcall sub_2CD5D60(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // r9
  __int64 v7; // r9
  __int64 result; // rax
  __int64 v9; // r12
  __int64 *v10; // r10
  __int64 v11; // r11
  __int64 v12; // r14
  __int64 v13; // rbx
  _QWORD *v14; // rsi
  __int64 v15; // rdx
  unsigned __int64 v16; // r8
  _QWORD *v17; // rcx
  int v18; // eax
  _QWORD *v19; // rdx
  unsigned __int64 v20; // r13
  _QWORD *v21; // rax
  unsigned __int64 v22; // rdi
  _QWORD *v23; // rdx
  int v24; // eax
  __int64 v25; // [rsp+8h] [rbp-88h]
  char v26; // [rsp+17h] [rbp-79h]
  __int64 v27; // [rsp+18h] [rbp-78h]
  __int64 *v28; // [rsp+18h] [rbp-78h]
  __int64 *v29; // [rsp+20h] [rbp-70h]
  int v30; // [rsp+20h] [rbp-70h]
  _QWORD *v31; // [rsp+28h] [rbp-68h]
  unsigned __int64 v32; // [rsp+38h] [rbp-58h] BYREF
  _QWORD v33[10]; // [rsp+40h] [rbp-50h] BYREF

  v5 = *(_QWORD *)(a3 + 40);
  if ( *(_BYTE *)a3 == 84 )
  {
    v7 = sub_AA4FF0(v5);
    if ( v7 )
      v7 -= 24;
  }
  else
  {
    v6 = *(_QWORD *)(a3 + 32);
    if ( !v6 || v6 == v5 + 48 )
      v7 = 0;
    else
      v7 = v6 - 24;
  }
  result = *(_QWORD *)(a2 + 24);
  v9 = *(_QWORD *)(result + 16);
  if ( v9 )
  {
    v10 = a1;
    v11 = a3;
    v12 = v7;
    do
    {
      v13 = *v10;
      v33[0] = v12;
      v14 = v33;
      v33[1] = v11;
      v15 = *(unsigned int *)(v13 + 8);
      v16 = *(unsigned int *)(v13 + 12);
      v33[2] = v9;
      v17 = *(_QWORD **)v13;
      v18 = v15;
      if ( v15 + 1 > v16 )
      {
        if ( v17 > v33 || v33 >= &v17[3 * v15] )
        {
          v26 = 0;
          v20 = -1;
        }
        else
        {
          v26 = 1;
          v20 = 0xAAAAAAAAAAAAAAABLL * (v33 - v17);
        }
        v27 = v11;
        v29 = v10;
        v17 = (_QWORD *)sub_C8D7D0(v13, v13 + 16, v15 + 1, 0x18u, &v32, v15 + 1);
        v21 = *(_QWORD **)v13;
        v10 = v29;
        v11 = v27;
        v22 = *(_QWORD *)v13 + 24LL * *(unsigned int *)(v13 + 8);
        if ( *(_QWORD *)v13 != v22 )
        {
          v23 = v17;
          do
          {
            if ( v23 )
            {
              *v23 = *v21;
              v23[1] = v21[1];
              v23[2] = v21[2];
            }
            v21 += 3;
            v23 += 3;
          }
          while ( (_QWORD *)v22 != v21 );
          v22 = *(_QWORD *)v13;
        }
        v24 = v32;
        if ( v13 + 16 != v22 )
        {
          v25 = v27;
          v28 = v29;
          v30 = v32;
          v31 = v17;
          _libc_free(v22);
          v11 = v25;
          v10 = v28;
          v24 = v30;
          v17 = v31;
        }
        v15 = *(unsigned int *)(v13 + 8);
        *(_DWORD *)(v13 + 12) = v24;
        v14 = v33;
        *(_QWORD *)v13 = v17;
        v18 = v15;
        if ( v26 )
          v14 = &v17[3 * v20];
      }
      v19 = &v17[3 * v15];
      if ( v19 )
      {
        *v19 = *v14;
        v19[1] = v14[1];
        v19[2] = v14[2];
        v18 = *(_DWORD *)(v13 + 8);
      }
      result = (unsigned int)(v18 + 1);
      *(_DWORD *)(v13 + 8) = result;
      v9 = *(_QWORD *)(v9 + 8);
    }
    while ( v9 );
  }
  return result;
}
