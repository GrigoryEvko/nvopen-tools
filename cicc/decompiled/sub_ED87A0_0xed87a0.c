// Function: sub_ED87A0
// Address: 0xed87a0
//
__int64 __fastcall sub_ED87A0(__int64 *a1, unsigned __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r9
  __int64 result; // rax
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 *v8; // rax
  _QWORD *v9; // r9
  __int64 v10; // rbx
  _QWORD *v11; // r15
  __int64 v12; // [rsp+0h] [rbp-50h]
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+10h] [rbp-40h]
  __int64 *v15; // [rsp+18h] [rbp-38h]
  _QWORD *v16; // [rsp+18h] [rbp-38h]

  if ( a2 > 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  v2 = *a1;
  v3 = a1[2] - *a1;
  result = v3 >> 5;
  if ( a2 > v3 >> 5 )
  {
    v5 = a1[1];
    v12 = 32 * a2;
    v13 = v5 - v2;
    if ( a2 )
    {
      v14 = sub_22077B0(32 * a2);
      if ( v2 == v5 )
      {
LABEL_16:
        v10 = a1[1];
        v5 = *a1;
        if ( v10 == *a1 )
        {
          v3 = a1[2] - v5;
        }
        else
        {
          do
          {
            v11 = *(_QWORD **)(v5 + 8);
            if ( v11 )
            {
              if ( (_QWORD *)*v11 != v11 + 2 )
                j_j___libc_free_0(*v11, v11[2] + 1LL);
              j_j___libc_free_0(v11, 32);
            }
            v5 += 32;
          }
          while ( v10 != v5 );
          v5 = *a1;
          v3 = a1[2] - *a1;
        }
        goto LABEL_23;
      }
    }
    else
    {
      v14 = 0;
      if ( v2 == v5 )
      {
LABEL_23:
        if ( v5 )
          j_j___libc_free_0(v5, v3);
        *a1 = v14;
        a1[1] = v14 + v13;
        a1[2] = v12 + v14;
        return v12 + v14;
      }
    }
    v6 = v14;
    do
    {
      if ( v6 )
      {
        *(_QWORD *)(v6 + 8) = 0;
        *(_DWORD *)(v6 + 16) = 0;
        *(_DWORD *)(v6 + 20) = 0;
        *(_BYTE *)(v6 + 24) = 0;
        *(_QWORD *)v6 = *(_QWORD *)v2;
        v7 = *(_QWORD *)(v2 + 8);
        if ( v7 )
        {
          v8 = (__int64 *)sub_22077B0(32);
          if ( v8 )
          {
            v15 = v8;
            *v8 = (__int64)(v8 + 2);
            sub_ED71E0(v8, *(_BYTE **)v7, *(_QWORD *)v7 + *(_QWORD *)(v7 + 8));
            v8 = v15;
          }
          v9 = *(_QWORD **)(v6 + 8);
          *(_QWORD *)(v6 + 8) = v8;
          if ( v9 )
          {
            if ( (_QWORD *)*v9 != v9 + 2 )
            {
              v16 = v9;
              j_j___libc_free_0(*v9, v9[2] + 1LL);
              v9 = v16;
            }
            j_j___libc_free_0(v9, 32);
          }
        }
        *(_DWORD *)(v6 + 16) = *(_DWORD *)(v2 + 16);
        *(_DWORD *)(v6 + 20) = *(_DWORD *)(v2 + 20);
        *(_BYTE *)(v6 + 24) = *(_BYTE *)(v2 + 24);
      }
      v2 += 32;
      v6 += 32;
    }
    while ( v5 != v2 );
    goto LABEL_16;
  }
  return result;
}
