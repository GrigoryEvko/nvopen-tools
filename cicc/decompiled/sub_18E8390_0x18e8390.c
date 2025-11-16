// Function: sub_18E8390
// Address: 0x18e8390
//
__int64 __fastcall sub_18E8390(__int64 *a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 *v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned int v7; // ecx
  __int64 *v8; // rbx
  __int64 v9; // r14
  __int64 v10; // rax
  unsigned int v11; // edx
  __int64 v12; // r9
  unsigned __int64 v13; // rsi
  __int64 v14; // r14
  __int64 v15; // r13
  unsigned int v16; // r12d
  unsigned int v17; // r15d
  __int64 v18; // rax
  unsigned int v19; // edx
  unsigned int v20; // eax
  int v23; // [rsp+Ch] [rbp-D4h]
  __int64 v25; // [rsp+20h] [rbp-C0h]
  __int64 v26; // [rsp+38h] [rbp-A8h]
  unsigned int v27; // [rsp+40h] [rbp-A0h]
  int v28; // [rsp+5Ch] [rbp-84h]
  __int64 *v30; // [rsp+68h] [rbp-78h]
  __int64 v31; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v32; // [rsp+78h] [rbp-68h]
  unsigned __int64 v33; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v34; // [rsp+88h] [rbp-58h]
  unsigned __int64 v35; // [rsp+90h] [rbp-50h]
  unsigned int v36; // [rsp+98h] [rbp-48h]
  char v37; // [rsp+A0h] [rbp-40h]

  v4 = a2;
  v5 = *(_QWORD *)(a1[3] + 56) + 112LL;
  if ( ((unsigned __int8)sub_1560180(v5, 34) || (unsigned __int8)sub_1560180(v5, 17))
    && (char *)a3 - (char *)a2 <= 16000 )
  {
    if ( a3 != a2 )
    {
      v30 = a2;
      v23 = -1;
      v27 = 0;
      while ( 1 )
      {
        v6 = v30[18];
        v32 = *(_DWORD *)(v6 + 32);
        if ( v32 > 0x40 )
          sub_16A4FD0((__int64)&v31, (const void **)(v6 + 24));
        else
          v31 = *(_QWORD *)(v6 + 24);
        v7 = *((_DWORD *)v30 + 2);
        v27 += v7;
        v25 = *v30 + 16LL * v7;
        if ( *v30 != v25 )
          break;
        v28 = 0;
LABEL_44:
        if ( v28 > v23 )
        {
          v23 = v28;
          *(_QWORD *)a4 = v30;
        }
        if ( v32 > 0x40 && v31 )
          j_j___libc_free_0_0(v31);
        v30 += 20;
        if ( a3 == v30 )
          return v27;
      }
      v26 = *v30;
      v28 = 0;
LABEL_9:
      v8 = a2;
      v28 += sub_14A30E0(*a1);
      while ( 1 )
      {
        v14 = v8[18];
        v15 = v30[18];
        v16 = *(_DWORD *)(v14 + 32);
        v17 = *(_DWORD *)(v15 + 32);
        if ( v16 <= 0x40 )
        {
          v9 = *(_QWORD *)(v14 + 24);
        }
        else
        {
          if ( v16 - (unsigned int)sub_16A57B0(v14 + 24) > 0x40 )
          {
            if ( v17 <= 0x40 )
              goto LABEL_22;
            v9 = -1;
LABEL_27:
            if ( v17 - (unsigned int)sub_16A57B0(v15 + 24) <= 0x40 )
            {
              v18 = **(_QWORD **)(v15 + 24);
              if ( v18 != -1 && v9 != -1 )
              {
                v19 = v17;
                v12 = v9 - v18;
                if ( v16 >= v17 )
                  v19 = v16;
                v34 = v19;
                goto LABEL_33;
              }
            }
            goto LABEL_22;
          }
          v9 = **(_QWORD **)(v14 + 24);
        }
        if ( v17 > 0x40 )
          goto LABEL_27;
        v10 = *(_QWORD *)(v15 + 24);
        if ( v9 != -1 && v10 != -1 )
        {
          v11 = v17;
          if ( v16 >= v17 )
            v11 = v16;
          v12 = v9 - v10;
          v34 = v11;
          if ( v11 <= 0x40 )
          {
            v13 = v12 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v11);
LABEL_18:
            v36 = v11;
            v35 = v13;
            v37 = 1;
            v28 -= sub_14A3080(*a1);
            if ( v37 && v36 > 0x40 && v35 )
              j_j___libc_free_0_0(v35);
            goto LABEL_22;
          }
LABEL_33:
          sub_16A4EF0((__int64)&v33, v12, 1);
          v11 = v34;
          v13 = v33;
          goto LABEL_18;
        }
LABEL_22:
        v8 += 20;
        if ( a3 == v8 )
        {
          v26 += 16;
          if ( v25 == v26 )
            goto LABEL_44;
          goto LABEL_9;
        }
      }
    }
    return 0;
  }
  if ( a3 == a2 )
    return 0;
  v20 = 0;
  do
  {
    v20 += *((_DWORD *)v4 + 2);
    if ( *((_DWORD *)v4 + 38) > *(_DWORD *)(*(_QWORD *)a4 + 152LL) )
      *(_QWORD *)a4 = v4;
    v4 += 20;
  }
  while ( a3 != v4 );
  return v20;
}
