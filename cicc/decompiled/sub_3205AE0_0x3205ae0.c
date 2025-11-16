// Function: sub_3205AE0
// Address: 0x3205ae0
//
__int64 __fastcall sub_3205AE0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 v7; // rdx
  unsigned __int8 *v8; // r8
  __int16 v10; // ax
  __int16 v11; // r14
  __int16 v12; // ax
  __int16 v13; // bx
  unsigned __int8 v14; // al
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned int v21; // r14d
  __int64 v22; // rax
  int v23; // edx
  int v24; // r9d
  unsigned __int64 v25[2]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v26; // [rsp+20h] [rbp-70h] BYREF
  _WORD v27[3]; // [rsp+30h] [rbp-60h] BYREF
  int v28; // [rsp+36h] [rbp-5Ah]
  unsigned __int64 v29; // [rsp+40h] [rbp-50h]
  unsigned __int64 v30; // [rsp+48h] [rbp-48h]
  __int64 v31; // [rsp+50h] [rbp-40h]
  __int64 v32; // [rsp+58h] [rbp-38h]
  int v33; // [rsp+60h] [rbp-30h]
  int v34; // [rsp+64h] [rbp-2Ch]
  __int64 v35; // [rsp+68h] [rbp-28h]

  if ( (unsigned __int8)sub_31F7430((__int64)a2) )
  {
    v4 = *(unsigned int *)(a1 + 1272);
    v5 = *(_QWORD *)(a1 + 1256);
    if ( (_DWORD)v4 )
    {
      v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v7 = v5 + 16LL * v6;
      v8 = *(unsigned __int8 **)v7;
      if ( a2 == *(unsigned __int8 **)v7 )
      {
LABEL_4:
        if ( v7 != v5 + 16 * v4 && !*(_DWORD *)(v7 + 8) )
          sub_C64ED0("cannot debug circular reference to unnamed type", 1u);
      }
      else
      {
        v23 = 1;
        while ( v8 != (unsigned __int8 *)-4096LL )
        {
          v24 = v23 + 1;
          v6 = (v4 - 1) & (v23 + v6);
          v7 = v5 + 16LL * v6;
          v8 = *(unsigned __int8 **)v7;
          if ( a2 == *(unsigned __int8 **)v7 )
            goto LABEL_4;
          v23 = v24;
        }
      }
    }
    return sub_32053D0(a1, (__int64)a2);
  }
  else
  {
    v10 = sub_AF18C0((__int64)a2);
    if ( v10 == 2 )
    {
      v11 = 5380;
    }
    else
    {
      if ( v10 != 19 )
        BUG();
      v11 = 5381;
    }
    v12 = sub_31F58C0((__int64)a2);
    LOBYTE(v12) = v12 | 0x80;
    v13 = v12;
    sub_3205740((__int64)v25, a1, a2);
    v14 = *(a2 - 16);
    if ( (v14 & 2) != 0 )
      v15 = *((_QWORD *)a2 - 4);
    else
      v15 = (__int64)&a2[-8 * ((v14 >> 2) & 0xF) - 16];
    v16 = *(_QWORD *)(v15 + 56);
    if ( v16 )
      v16 = sub_B91420(v16);
    else
      v17 = 0;
    v31 = v16;
    v27[1] = 0;
    v27[0] = v11;
    v29 = v25[0];
    v27[2] = v13;
    v28 = 0;
    v30 = v25[1];
    v32 = v17;
    v33 = 0;
    v34 = 0;
    v35 = 0;
    v18 = sub_3709F10(a1 + 648, v27);
    v21 = sub_3707F80(a1 + 632, v18);
    if ( (a2[20] & 4) == 0 )
    {
      v22 = *(unsigned int *)(a1 + 1288);
      if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1292) )
      {
        sub_C8D5F0(a1 + 1280, (const void *)(a1 + 1296), v22 + 1, 8u, v19, v20);
        v22 = *(unsigned int *)(a1 + 1288);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 1280) + 8 * v22) = a2;
      ++*(_DWORD *)(a1 + 1288);
    }
    if ( (__int64 *)v25[0] != &v26 )
      j_j___libc_free_0(v25[0]);
    return v21;
  }
}
