// Function: sub_2416BC0
// Address: 0x2416bc0
//
__int64 __fastcall sub_2416BC0(__int64 *a1, __int64 a2)
{
  int v2; // ebx
  unsigned int v3; // ebx
  unsigned __int64 v4; // r13
  char *v5; // rax
  char *v6; // r15
  char *v7; // rax
  char *v8; // r15
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // r12
  char *v14; // [rsp+0h] [rbp-70h] BYREF
  char *v15; // [rsp+8h] [rbp-68h]
  char *v16; // [rsp+10h] [rbp-60h]
  unsigned __int64 v17; // [rsp+20h] [rbp-50h] BYREF
  char *v18; // [rsp+28h] [rbp-48h]
  char *v19; // [rsp+30h] [rbp-40h]

  v2 = *(_DWORD *)(a2 + 4);
  v14 = 0;
  v15 = 0;
  v3 = v2 & 0x7FFFFFF;
  v16 = 0;
  if ( v3 )
  {
    v4 = 8LL * v3;
    v5 = (char *)sub_22077B0(v4);
    v6 = &v5[v4];
    v14 = v5;
    v16 = &v5[v4];
    memset(v5, 0, v4);
    v15 = v6;
    v7 = (char *)sub_22077B0(v4);
    v8 = &v7[v4];
    v17 = (unsigned __int64)v7;
    v19 = &v7[v4];
    memset(v7, 0, v4);
    v18 = v8;
    v9 = 0;
    do
    {
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v10 = *(_QWORD *)(a2 - 8);
      else
        v10 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      *(_QWORD *)&v14[v9] = sub_24159D0((__int64)a1, *(_QWORD *)(v10 + 4 * v9));
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v11 = *(_QWORD *)(a2 - 8);
      else
        v11 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      *(_QWORD *)(v17 + v9) = sub_2414930(a1, *(_BYTE **)(v11 + 4 * v9));
      v9 += 8;
    }
    while ( v4 != v9 );
  }
  else
  {
    v17 = 0;
    v19 = 0;
    v18 = 0;
  }
  v12 = sub_2415600((__int64)a1, &v14, (__int64 *)&v17, a2 + 24, 0, 0);
  if ( v17 )
    j_j___libc_free_0(v17);
  if ( v14 )
    j_j___libc_free_0((unsigned __int64)v14);
  return v12;
}
