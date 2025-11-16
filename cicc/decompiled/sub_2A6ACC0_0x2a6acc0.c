// Function: sub_2A6ACC0
// Address: 0x2a6acc0
//
void __fastcall sub_2A6ACC0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // rax
  unsigned int v3; // r13d
  __int64 v4; // rbx
  __int64 v5; // rax
  unsigned int v6; // edx
  _QWORD *v7; // rdx
  unsigned __int8 v8; // al
  __int64 v9; // rbx
  __int64 v10; // rbx
  _QWORD *v11; // rax
  unsigned int v12; // edx
  unsigned int v13; // [rsp+0h] [rbp-B0h]
  _QWORD *v14; // [rsp+0h] [rbp-B0h]
  _QWORD *v15; // [rsp+0h] [rbp-B0h]
  __int64 v16; // [rsp+18h] [rbp-98h]
  unsigned __int8 v17[48]; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int8 v18; // [rsp+50h] [rbp-60h] BYREF
  char v19; // [rsp+51h] [rbp-5Fh]
  unsigned __int64 v20; // [rsp+58h] [rbp-58h] BYREF
  unsigned int v21; // [rsp+60h] [rbp-50h]
  unsigned __int64 v22; // [rsp+68h] [rbp-48h] BYREF
  unsigned int v23; // [rsp+70h] [rbp-40h]

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 15 )
    goto LABEL_25;
  if ( *(_BYTE *)sub_2A68BC0(a1, (unsigned __int8 *)a2) == 6 )
    return;
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) > 0x40 )
  {
LABEL_25:
    sub_2A6A450(a1, a2);
    return;
  }
  v2 = (unsigned __int8 *)sub_2A68BC0(a1, (unsigned __int8 *)a2);
  sub_22C05A0((__int64)v17, v2);
  v3 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( v3 )
  {
    v4 = 0;
    v5 = 8LL * v3;
    v3 = 0;
    v16 = v5;
    while ( 1 )
    {
      if ( (unsigned __int8)sub_2A63A20(
                              a1,
                              *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL * *(unsigned int *)(a2 + 72) + v4),
                              *(_QWORD *)(a2 + 40)) )
      {
        v7 = sub_2A68BC0(a1, *(unsigned __int8 **)(*(_QWORD *)(a2 - 8) + 4 * v4));
        v8 = *(_BYTE *)v7;
        v19 = 0;
        v18 = v8;
        if ( v8 <= 3u )
        {
          if ( v8 > 1u )
            v20 = v7[1];
        }
        else if ( (unsigned __int8)(v8 - 4) <= 1u )
        {
          v21 = *((_DWORD *)v7 + 4);
          if ( v21 > 0x40 )
          {
            v15 = v7;
            sub_C43780((__int64)&v20, (const void **)v7 + 1);
            v7 = v15;
          }
          else
          {
            v20 = v7[1];
          }
          v23 = *((_DWORD *)v7 + 8);
          if ( v23 > 0x40 )
          {
            v14 = v7;
            sub_C43780((__int64)&v22, (const void **)v7 + 3);
            v7 = v14;
          }
          else
          {
            v22 = v7[3];
          }
          v19 = *((_BYTE *)v7 + 1);
        }
        sub_2A625F0((__int64)v17, (__int64)&v18, 0, 0, 1u);
        v6 = v3 + 1;
        if ( v17[0] == 6 )
        {
          v9 = v3 + 2;
          sub_22C0090(&v18);
          ++v3;
          goto LABEL_21;
        }
        if ( (unsigned int)v18 - 4 <= 1 )
        {
          if ( v23 > 0x40 && v22 )
          {
            j_j___libc_free_0_0(v22);
            v6 = v3 + 1;
          }
          if ( v21 > 0x40 && v20 )
          {
            v13 = v6;
            j_j___libc_free_0_0(v20);
            v6 = v13;
          }
        }
        v3 = v6;
      }
      v4 += 8;
      if ( v16 == v4 )
      {
        v9 = v3 + 1;
        goto LABEL_21;
      }
    }
  }
  v9 = 1;
LABEL_21:
  v10 = v9 << 32;
  BYTE1(v10) |= 1u;
  sub_22C05A0((__int64)&v18, v17);
  sub_2A689D0(a1, a2, &v18, v10);
  sub_22C0090(&v18);
  v11 = sub_2A68BC0(a1, (unsigned __int8 *)a2);
  v12 = *((unsigned __int8 *)v11 + 1);
  if ( v12 < v3 )
    LOBYTE(v12) = v3;
  *((_BYTE *)v11 + 1) = v12;
  sub_22C0090(v17);
}
