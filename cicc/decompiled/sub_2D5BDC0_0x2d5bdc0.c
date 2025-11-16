// Function: sub_2D5BDC0
// Address: 0x2d5bdc0
//
__int64 __fastcall sub_2D5BDC0(__int64 a1, __int64 a2)
{
  __int64 *v4; // rax
  _BYTE *v5; // r12
  __int64 v6; // r14
  __int64 *v7; // r15
  unsigned int v8; // eax
  __int64 v9; // rdx
  unsigned __int16 v10; // ax
  __int64 v11; // rax
  char v12; // si
  unsigned __int64 v13; // rax
  unsigned int v14; // eax
  unsigned int v15; // r8d
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // r10
  __int64 v20; // rcx
  __int64 (*v21)(); // rax
  bool v22; // zf
  int v23; // eax
  int v24; // ebx
  char v25; // al
  int v26; // eax
  _QWORD *v27; // r12
  __int64 v28; // rsi
  __int64 v29; // rbx
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rbx
  const void **v36; // rsi
  __int64 v37; // rsi
  unsigned __int8 *v38; // rsi
  char v39; // al
  unsigned int v40; // [rsp+8h] [rbp-88h]
  __int64 v41; // [rsp+10h] [rbp-80h]
  unsigned __int16 v42; // [rsp+18h] [rbp-78h]
  __int64 v43; // [rsp+18h] [rbp-78h]
  __int64 v44; // [rsp+18h] [rbp-78h]
  __int64 v45; // [rsp+18h] [rbp-78h]
  unsigned int v46; // [rsp+24h] [rbp-6Ch]
  int v47; // [rsp+28h] [rbp-68h]
  unsigned __int64 v48; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v49; // [rsp+38h] [rbp-58h]
  __int16 v50; // [rsp+50h] [rbp-40h]

  v4 = *(__int64 **)(a2 - 8);
  v5 = (_BYTE *)*v4;
  v6 = *(_QWORD *)(*v4 + 8);
  v7 = (__int64 *)sub_BD5C60(*v4);
  v8 = sub_2D5BAE0(*(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 816), (__int64 *)v6, 0);
  v41 = v9;
  v40 = v8;
  v10 = (*(__int64 (__fastcall **)(_QWORD, __int64 *, _QWORD, __int64))(**(_QWORD **)(a1 + 16) + 680LL))(
          *(_QWORD *)(a1 + 16),
          v7,
          v8,
          v9);
  if ( v10 <= 1u || (unsigned __int16)(v10 - 504) <= 7u )
    BUG();
  v42 = v10;
  v11 = 16LL * (v10 - 1);
  v12 = byte_444C4A0[v11 + 8];
  v13 = *(_QWORD *)&byte_444C4A0[v11];
  LOBYTE(v49) = v12;
  v48 = v13;
  v14 = sub_CA1930(&v48);
  v15 = 0;
  v46 = v14;
  if ( v14 > *(_DWORD *)(v6 + 8) >> 8 )
  {
    v17 = sub_BCD140(v7, v14);
    v18 = *(_QWORD *)(a1 + 16);
    v19 = v17;
    v20 = v42;
    v21 = *(__int64 (**)())(*(_QWORD *)v18 + 1456LL);
    if ( v21 == sub_2D56680 )
    {
      v47 = 39;
    }
    else
    {
      v45 = v19;
      v39 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, __int64, _QWORD))v21)(v18, v40, v41, v20, 0);
      v19 = v45;
      v47 = 39 - ((v39 == 0) - 1);
    }
    if ( *v5 == 22 )
    {
      v43 = v19;
      v22 = (unsigned __int8)sub_B2D770((__int64)v5) == 0;
      v23 = 40;
      if ( v22 )
        v23 = v47;
      v24 = v23;
      v25 = sub_B2D760((__int64)v5);
      v19 = v43;
      v22 = v25 == 0;
      v26 = 39;
      if ( v22 )
        v26 = v24;
      v47 = v26;
    }
    v50 = 257;
    v27 = (_QWORD *)sub_B51D30(v47, (__int64)v5, v19, (__int64)&v48, 0, 0);
    sub_B44220(v27, a2 + 24, 0);
    v28 = *(_QWORD *)(a2 + 48);
    v48 = v28;
    if ( v28 )
    {
      v29 = (__int64)(v27 + 6);
      sub_B96E90((__int64)&v48, v28, 1);
      if ( v27 + 6 == &v48 )
      {
        if ( v48 )
          sub_B91220((__int64)&v48, v48);
        goto LABEL_17;
      }
      v37 = v27[6];
      if ( !v37 )
      {
LABEL_37:
        v38 = (unsigned __int8 *)v48;
        v27[6] = v48;
        if ( v38 )
          sub_B976B0((__int64)&v48, v38, v29);
        goto LABEL_17;
      }
    }
    else
    {
      v29 = (__int64)(v27 + 6);
      if ( v27 + 6 == &v48 || (v37 = v27[6]) == 0 )
      {
LABEL_17:
        sub_AC2B30(*(_QWORD *)(a2 - 8), (__int64)v27);
        v44 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
        if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1 != 1 )
        {
          v30 = 0;
          do
          {
            v35 = 32LL * (unsigned int)(2 * ++v30);
            v36 = (const void **)(*(_QWORD *)(*(_QWORD *)(a2 - 8) + v35) + 24LL);
            if ( v47 == 39 )
              sub_C449B0((__int64)&v48, v36, v46);
            else
              sub_C44830((__int64)&v48, v36, v46);
            v31 = sub_ACCFD0(v7, (__int64)&v48);
            v32 = *(_QWORD *)(a2 - 8) + v35;
            if ( *(_QWORD *)v32 )
            {
              v33 = *(_QWORD *)(v32 + 8);
              **(_QWORD **)(v32 + 16) = v33;
              if ( v33 )
                *(_QWORD *)(v33 + 16) = *(_QWORD *)(v32 + 16);
            }
            *(_QWORD *)v32 = v31;
            if ( v31 )
            {
              v34 = *(_QWORD *)(v31 + 16);
              *(_QWORD *)(v32 + 8) = v34;
              if ( v34 )
                *(_QWORD *)(v34 + 16) = v32 + 8;
              *(_QWORD *)(v32 + 16) = v31 + 16;
              *(_QWORD *)(v31 + 16) = v32;
            }
            if ( v49 > 0x40 && v48 )
              j_j___libc_free_0_0(v48);
          }
          while ( v44 != v30 );
        }
        return 1;
      }
    }
    sub_B91220(v29, v37);
    goto LABEL_37;
  }
  return v15;
}
