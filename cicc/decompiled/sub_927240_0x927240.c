// Function: sub_927240
// Address: 0x927240
//
__int64 __fastcall sub_927240(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rcx
  __int64 v7; // r15
  __int64 *v8; // rax
  unsigned __int64 v9; // rsi
  unsigned int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // r11
  __int64 v15; // rdi
  __int64 (__fastcall *v16)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v17; // rax
  __int64 v18; // r13
  _BOOL4 v19; // edx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  char v23; // al
  __int64 v24; // rdx
  _BOOL4 v25; // edx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v29; // rdx
  unsigned int v30; // r15d
  unsigned int *v31; // rax
  unsigned int *v32; // rbx
  unsigned int *v33; // r15
  __int64 v34; // rdx
  __int64 v35; // rsi
  unsigned int v36; // ecx
  __int64 v37; // rax
  unsigned int v38; // eax
  unsigned int v39; // eax
  unsigned int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // [rsp+8h] [rbp-E8h]
  __int64 v45; // [rsp+8h] [rbp-E8h]
  __int64 v46; // [rsp+8h] [rbp-E8h]
  __int64 v47; // [rsp+10h] [rbp-E0h]
  __int64 v48; // [rsp+10h] [rbp-E0h]
  unsigned int v49; // [rsp+1Ch] [rbp-D4h]
  const char *v50; // [rsp+20h] [rbp-D0h] BYREF
  char v51; // [rsp+40h] [rbp-B0h]
  char v52; // [rsp+41h] [rbp-AFh]
  char v53[32]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v54; // [rsp+70h] [rbp-80h]
  int v55; // [rsp+80h] [rbp-70h] BYREF
  __int64 v56; // [rsp+88h] [rbp-68h]
  unsigned int v57; // [rsp+98h] [rbp-58h]
  __int64 v58; // [rsp+A0h] [rbp-50h]
  __int64 v59; // [rsp+A8h] [rbp-48h]

  v47 = *(_QWORD *)(a3 + 72);
  sub_926800((__int64)&v55, a2, v47);
  v7 = v56;
  v49 = v57;
  v8 = (__int64 *)v47;
  if ( v55 != 1 )
  {
    v9 = *(_QWORD *)(a3 + 8);
    v48 = v9;
    v10 = *(_DWORD *)(*(_QWORD *)(v56 + 8) + 8LL) >> 8;
    if ( v9 )
    {
      v11 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, v9, 0, v6);
      v12 = sub_BCE760(v11, v10);
      v13 = *(_QWORD *)(a3 + 8);
      v14 = v12;
      if ( *(char *)(v13 + 142) < 0 )
      {
LABEL_5:
        v49 = *(_DWORD *)(v13 + 136);
        goto LABEL_6;
      }
    }
    else
    {
      if ( (*(_BYTE *)(a3 + 25) & 1) == 0 && sub_8D2310(*v8) )
      {
        if ( (unsigned int)sub_8D2E30(*(_QWORD *)a3) )
        {
          v41 = sub_8D46C0(*(_QWORD *)a3);
          if ( sub_8D2310(v41) )
          {
            v48 = sub_8D46C0(*(_QWORD *)a3);
            v43 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *(_QWORD *)a3, 0, v42);
            v13 = *(_QWORD *)a3;
            v14 = v43;
            goto LABEL_6;
          }
        }
      }
      v48 = *(_QWORD *)a3;
      v20 = sub_91A390(*(_QWORD *)(a2 + 32) + 8LL, *(_QWORD *)a3, 0, v6);
      v21 = sub_BCE760(v20, v10);
      v13 = *(_QWORD *)a3;
      v14 = v21;
      if ( *(char *)(*(_QWORD *)a3 + 142LL) < 0 )
        goto LABEL_5;
    }
    if ( *(_BYTE *)(v13 + 140) != 12 )
      goto LABEL_5;
    v46 = v14;
    v38 = sub_8D4AB0(v13);
    v14 = v46;
    v49 = v38;
LABEL_6:
    v52 = 1;
    v50 = "lvaladjust";
    v51 = 3;
    if ( v14 == *(_QWORD *)(v7 + 8) )
    {
      v18 = v7;
      goto LABEL_13;
    }
    v15 = *(_QWORD *)(a2 + 128);
    v16 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v15 + 120LL);
    if ( v16 == sub_920130 )
    {
      if ( *(_BYTE *)v7 > 0x15u )
        goto LABEL_30;
      v44 = v14;
      if ( (unsigned __int8)sub_AC4810(49) )
        v17 = sub_ADAB70(49, v7, v44, 0);
      else
        v17 = sub_AA93C0(49, v7, v44);
      v14 = v44;
      v18 = v17;
    }
    else
    {
      v45 = v14;
      v37 = v16(v15, 49u, (_BYTE *)v7, v14);
      v14 = v45;
      v18 = v37;
    }
    if ( v18 )
    {
LABEL_13:
      v19 = 0;
      if ( (*(_BYTE *)(v13 + 140) & 0xFB) == 8 )
        v19 = (sub_8D4C10(v13, dword_4F077C4 != 2) & 2) != 0;
      *(_DWORD *)a1 = 0;
      *(_QWORD *)(a1 + 8) = v18;
      *(_QWORD *)(a1 + 16) = v48;
      *(_DWORD *)(a1 + 48) = v19;
      *(_DWORD *)(a1 + 24) = v49;
      return a1;
    }
LABEL_30:
    v54 = 257;
    v18 = sub_B51D30(49, v7, v14, v53, 0, 0);
    if ( (unsigned __int8)sub_920620(v18) )
    {
      v29 = *(_QWORD *)(a2 + 144);
      v30 = *(_DWORD *)(a2 + 152);
      if ( v29 )
        sub_B99FD0(v18, 3, v29);
      sub_B45150(v18, v30);
    }
    (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
      *(_QWORD *)(a2 + 136),
      v18,
      &v50,
      *(_QWORD *)(a2 + 104),
      *(_QWORD *)(a2 + 112));
    v31 = *(unsigned int **)(a2 + 48);
    v32 = &v31[4 * *(unsigned int *)(a2 + 56)];
    if ( v31 != v32 )
    {
      v33 = v31;
      do
      {
        v34 = *((_QWORD *)v33 + 1);
        v35 = *v33;
        v33 += 4;
        sub_B99FD0(v18, v35, v34);
      }
      while ( v32 != v33 );
    }
    goto LABEL_13;
  }
  v22 = *(_QWORD *)a3;
  v23 = *(_BYTE *)(*(_QWORD *)a3 + 140LL);
  v24 = *(_QWORD *)a3;
  if ( *(char *)(*(_QWORD *)a3 + 142LL) < 0 )
  {
    v36 = *(_DWORD *)(v22 + 136);
    if ( v57 >= v36 )
      goto LABEL_25;
    goto LABEL_39;
  }
  if ( v23 == 12 )
  {
    v39 = sub_8D4AB0(v22);
    v22 = *(_QWORD *)a3;
    if ( v49 >= v39 )
      goto LABEL_47;
    v24 = *(_QWORD *)a3;
    if ( *(char *)(v22 + 142) >= 0 )
    {
      if ( *(_BYTE *)(v22 + 140) != 12 )
        goto LABEL_24;
      v40 = sub_8D4AB0(v22);
      v22 = *(_QWORD *)a3;
      v49 = v40;
LABEL_47:
      v23 = *(_BYTE *)(v22 + 140);
      goto LABEL_25;
    }
    v36 = *(_DWORD *)(v22 + 136);
LABEL_39:
    v49 = v36;
    v23 = *(_BYTE *)(v24 + 140);
    goto LABEL_25;
  }
  if ( v57 < *(_DWORD *)(v22 + 136) )
  {
LABEL_24:
    v22 = v24;
    v49 = *(_DWORD *)(v24 + 136);
    v23 = *(_BYTE *)(v24 + 140);
  }
LABEL_25:
  v25 = 0;
  if ( (v23 & 0xFB) == 8 )
    v25 = (sub_8D4C10(v22, dword_4F077C4 != 2) & 2) != 0;
  v26 = v58;
  *(_DWORD *)a1 = 1;
  *(_QWORD *)(a1 + 8) = v7;
  *(_QWORD *)(a1 + 32) = v26;
  v27 = v59;
  *(_DWORD *)(a1 + 48) = v25;
  *(_QWORD *)(a1 + 40) = v27;
  *(_DWORD *)(a1 + 24) = v49;
  return a1;
}
