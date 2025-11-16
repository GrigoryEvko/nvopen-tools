// Function: sub_2C1CD70
// Address: 0x2c1cd70
//
void __fastcall sub_2C1CD70(__int64 a1, __int64 a2)
{
  unsigned int v4; // r10d
  __int64 v5; // r14
  __int64 *v6; // r8
  __int64 *v7; // r15
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // rdx
  unsigned __int64 v11; // r9
  __int64 v12; // rcx
  _BYTE *v13; // rdx
  unsigned __int8 *v14; // rax
  unsigned __int8 *v15; // r13
  _BYTE *v16; // rdx
  _QWORD *v17; // rax
  __int64 v18; // r15
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // r15
  _BYTE *v22; // rax
  __int64 v23; // rcx
  _BYTE *v24; // rdi
  int v25; // eax
  char v26; // al
  unsigned int v27; // esi
  _BYTE *v28; // r13
  _BYTE *v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // rax
  bool v32; // cc
  _QWORD *v33; // rax
  __int64 v34; // rax
  __int64 v35; // [rsp+0h] [rbp-C0h]
  __int64 *v36; // [rsp+18h] [rbp-A8h]
  __int64 v37; // [rsp+18h] [rbp-A8h]
  int v38; // [rsp+18h] [rbp-A8h]
  _BYTE *v39; // [rsp+18h] [rbp-A8h]
  __int64 v40; // [rsp+28h] [rbp-98h]
  _BYTE *v41; // [rsp+30h] [rbp-90h] BYREF
  __int64 v42; // [rsp+38h] [rbp-88h]
  _BYTE v43[16]; // [rsp+40h] [rbp-80h] BYREF
  __int16 v44; // [rsp+50h] [rbp-70h]
  __int64 v45[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v46; // [rsp+80h] [rbp-40h]

  v45[0] = *(_QWORD *)(a1 + 88);
  if ( v45[0] )
    sub_2AAAFA0(v45);
  sub_2BF1A90(a2, (__int64)v45);
  sub_9C6650(v45);
  v4 = *(_DWORD *)(a1 + 160);
  v5 = *(_QWORD *)(a2 + 904);
  if ( v4 == 64 )
  {
    v30 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 0);
    v31 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL) + 40LL);
    v46 = 257;
    v32 = *(_DWORD *)(v31 + 32) <= 0x40u;
    v33 = *(_QWORD **)(v31 + 24);
    if ( !v32 )
      v33 = (_QWORD *)*v33;
    LODWORD(v41) = (_DWORD)v33;
    v34 = sub_94D3D0((unsigned int **)v5, v30, (__int64)&v41, 1, (__int64)v45);
    v19 = a1 + 96;
    v20 = v34;
    goto LABEL_25;
  }
  if ( v4 > 0x40 )
  {
    if ( v4 != 67 )
      goto LABEL_42;
    v37 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 0);
    v44 = 257;
    v46 = 257;
    v17 = sub_BD2C40(72, 1u);
    v18 = (__int64)v17;
    if ( v17 )
      sub_B549F0((__int64)v17, v37, (__int64)v45, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(v5 + 88) + 16LL))(
      *(_QWORD *)(v5 + 88),
      v18,
      &v41,
      *(_QWORD *)(v5 + 56),
      *(_QWORD *)(v5 + 64));
    sub_94AAF0((unsigned int **)v5, v18);
    v19 = a1 + 96;
    v20 = v18;
LABEL_25:
    sub_2BF26E0(a2, v19, v20, 0);
    return;
  }
  if ( v4 <= 0x1E )
  {
    if ( v4 > 0xB )
    {
      v6 = *(__int64 **)(a1 + 48);
      v41 = v43;
      v42 = 0x200000000LL;
      v36 = &v6[*(unsigned int *)(a1 + 56)];
      if ( v36 == v6 )
      {
        v13 = v43;
        v12 = 0;
      }
      else
      {
        v7 = v6;
        do
        {
          v8 = sub_2BFB640(a2, *v7, 0);
          v10 = (unsigned int)v42;
          v11 = (unsigned int)v42 + 1LL;
          if ( v11 > HIDWORD(v42) )
          {
            v35 = v8;
            sub_C8D5F0((__int64)&v41, v43, (unsigned int)v42 + 1LL, 8u, v9, v11);
            v10 = (unsigned int)v42;
            v8 = v35;
          }
          ++v7;
          *(_QWORD *)&v41[8 * v10] = v8;
          v12 = (unsigned int)(v42 + 1);
          LODWORD(v42) = v42 + 1;
        }
        while ( v36 != v7 );
        v4 = *(_DWORD *)(a1 + 160);
        v13 = v41;
      }
      v46 = 257;
      v14 = (unsigned __int8 *)sub_B356A0((unsigned int **)v5, v4, v13, v12, (__int64)v45, 0);
      v15 = v14;
      if ( *v14 > 0x1Cu )
        sub_2AAF930(a1, v14);
      sub_2BF26E0(a2, a1 + 96, (__int64)v15, 0);
      v16 = *(_BYTE **)(a1 + 136);
      if ( v16 && *v16 <= 0x1Cu )
        v16 = 0;
      sub_2BF08A0(a2, v15, v16);
      if ( v41 != v43 )
        _libc_free((unsigned __int64)v41);
      return;
    }
LABEL_42:
    BUG();
  }
  if ( v4 - 53 > 1 )
    goto LABEL_42;
  v38 = *(_DWORD *)(a1 + 160);
  v21 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 0);
  v22 = (_BYTE *)sub_2BFB640(a2, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL), 0);
  v46 = 257;
  v23 = (__int64)v22;
  if ( v38 == 54 )
  {
    v24 = *(_BYTE **)(a1 + 136);
    if ( v24 && *v24 > 0x1Cu )
    {
      v39 = v22;
      v25 = sub_B45210((__int64)v24);
      v23 = (__int64)v39;
      LODWORD(v40) = v25;
      v26 = 1;
    }
    else
    {
      v26 = 0;
    }
    BYTE4(v40) = v26;
    v27 = *(_DWORD *)(a1 + 156);
    v41 = (_BYTE *)v40;
    v28 = (_BYTE *)sub_B35C90(v5, v27, v21, v23, (__int64)v45, 0, v40, 0);
  }
  else
  {
    v28 = (_BYTE *)sub_92B530((unsigned int **)v5, *(_DWORD *)(a1 + 156), v21, v22, (__int64)v45);
  }
  sub_2BF26E0(a2, a1 + 96, (__int64)v28, 0);
  v29 = *(_BYTE **)(a1 + 136);
  if ( v29 && *v29 <= 0x1Cu )
    v29 = 0;
  sub_2BF08A0(a2, v28, v29);
}
