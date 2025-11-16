// Function: sub_F4B700
// Address: 0xf4b700
//
char __fastcall sub_F4B700(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        _BYTE *a6,
        _BYTE *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v13; // rax
  __int64 v14; // r14
  unsigned __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v19; // r12
  _QWORD *v20; // rdi
  __int64 v21; // r9
  __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // rcx
  _QWORD *v25; // rdi
  __int64 v26; // rax
  __int64 i; // rbx
  _QWORD *v28; // r11
  _QWORD *v29; // rdx
  _QWORD *v30; // r10
  __int64 v31; // r13
  __int64 v32; // rdi
  unsigned int v34; // [rsp+4h] [rbp-9Ch]
  __int64 v35; // [rsp+10h] [rbp-90h]
  __int64 v36; // [rsp+18h] [rbp-88h]
  __int64 v37; // [rsp+18h] [rbp-88h]
  __int64 v38; // [rsp+18h] [rbp-88h]
  __int64 v39; // [rsp+20h] [rbp-80h]
  __int64 v41; // [rsp+30h] [rbp-70h]
  _QWORD *v42; // [rsp+30h] [rbp-70h]
  _QWORD *v44; // [rsp+38h] [rbp-68h]
  __int64 v45[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v46; // [rsp+60h] [rbp-40h]

  LOBYTE(v13) = sub_B2FC80(a2);
  if ( !(_BYTE)v13 )
  {
    v41 = a2 + 72;
    if ( *(_QWORD *)(a2 + 80) == a2 + 72 )
      goto LABEL_33;
    v34 = a4;
    v14 = *(_QWORD *)(a2 + 80);
    do
    {
      v18 = 0;
      v46 = 257;
      if ( v14 )
        v18 = v14 - 24;
      if ( *a6 )
      {
        v45[0] = (__int64)a6;
        LOBYTE(v46) = 3;
      }
      v19 = sub_F4B360(v18, a3, v45, a1, a7);
      v20 = sub_F46C80(a3, v18);
      v22 = v20[2];
      if ( v19 != v22 )
      {
        if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
          sub_BD60C0(v20);
        v20[2] = v19;
        if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
          sub_BD73F0((__int64)v20);
      }
      if ( (*(_WORD *)(v18 + 2) & 0x7FFF) != 0 )
      {
        v35 = sub_ACC1C0(a2, v18);
        v36 = sub_ACC1C0(a1, v19);
        v23 = sub_F46C80(a3, v35);
        v24 = v36;
        v25 = v23;
        v26 = v23[2];
        if ( v36 != v26 )
        {
          if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
          {
            sub_BD60C0(v25);
            v24 = v36;
          }
          v25[2] = v24;
          if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
            sub_BD73F0((__int64)v25);
        }
      }
      v15 = *(_QWORD *)(v19 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v15 == v19 + 48 )
        goto LABEL_46;
      if ( !v15 )
        BUG();
      v16 = v15 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 > 0xA )
LABEL_46:
        BUG();
      if ( *(_BYTE *)(v15 - 24) == 30 )
      {
        v17 = *(unsigned int *)(a5 + 8);
        if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
        {
          v38 = v16;
          sub_C8D5F0(a5, (const void *)(a5 + 16), v17 + 1, 8u, v16, v21);
          v17 = *(unsigned int *)(a5 + 8);
          v16 = v38;
        }
        *(_QWORD *)(*(_QWORD *)a5 + 8 * v17) = v16;
        ++*(_DWORD *)(a5 + 8);
      }
      v14 = *(_QWORD *)(v14 + 8);
    }
    while ( v41 != v14 );
    a4 = v34;
    v41 = *(_QWORD *)(a2 + 80);
    if ( v41 )
LABEL_33:
      v41 -= 24;
    v39 = sub_F46C80(a3, v41)[2] + 24LL;
    LOBYTE(v13) = a1 + 72;
    v37 = a1 + 72;
    if ( v39 != a1 + 72 )
    {
      while ( 1 )
      {
        for ( i = *(_QWORD *)(v39 + 32); v39 + 24 != i; i = *(_QWORD *)(i + 8) )
        {
          if ( !i )
          {
            sub_FC75A0(v45, a3, a4, a8, a9, a10);
            sub_FCD280(v45, 0);
            sub_FC7680(v45);
            BUG();
          }
          sub_FC75A0(v45, a3, a4, a8, a9, a10);
          sub_FCD280(v45, i - 24);
          sub_FC7680(v45);
          v32 = *(_QWORD *)(i + 40);
          if ( v32 )
          {
            v28 = (_QWORD *)sub_B14240(v32);
            v30 = v29;
          }
          else
          {
            v30 = &qword_4F81430[1];
            v28 = &qword_4F81430[1];
          }
          v42 = v28;
          v44 = v30;
          v31 = sub_B43CA0(i - 24);
          sub_FC75A0(v45, a3, a4, a8, a9, a10);
          sub_FCD310(v45, v31, v42, v44);
          sub_FC7680(v45);
        }
        v13 = *(_QWORD *)(v39 + 8);
        v39 = v13;
        if ( v37 == v13 )
          break;
        if ( !v13 )
          BUG();
      }
    }
  }
  return v13;
}
