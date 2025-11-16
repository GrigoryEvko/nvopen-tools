// Function: sub_2C1E6D0
// Address: 0x2c1e6d0
//
__int64 __fastcall sub_2C1E6D0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rax
  int v7; // ecx
  __int64 v8; // r15
  __int64 v9; // rsi
  char v10; // dl
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned int **v16; // rdi
  unsigned int **v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v21; // rax
  __int64 v22; // r10
  int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r11
  int v27; // edi
  int v28; // eax
  __int64 v29; // [rsp+0h] [rbp-A0h]
  __int64 v30; // [rsp+8h] [rbp-98h]
  __int64 v31; // [rsp+8h] [rbp-98h]
  __int64 v32; // [rsp+10h] [rbp-90h]
  __int64 v33; // [rsp+10h] [rbp-90h]
  int v34; // [rsp+18h] [rbp-88h]
  __int64 v35; // [rsp+18h] [rbp-88h]
  __int64 v36; // [rsp+18h] [rbp-88h]
  int v37; // [rsp+20h] [rbp-80h]
  __int16 v38; // [rsp+24h] [rbp-7Ch]
  unsigned __int8 v39; // [rsp+27h] [rbp-79h]
  __int64 v40; // [rsp+28h] [rbp-78h]
  __int64 v41; // [rsp+30h] [rbp-70h]
  unsigned int v42; // [rsp+38h] [rbp-68h]
  __int64 v43[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v44; // [rsp+60h] [rbp-40h]

  v4 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 1);
  v5 = *(_QWORD *)(a2 + 904);
  v32 = v4;
  v6 = *(_QWORD *)(a1 + 152);
  v34 = *(_DWORD *)(v6 + 40);
  v7 = *(_DWORD *)(v5 + 104);
  *(_DWORD *)(v5 + 104) = *(_DWORD *)(v6 + 44);
  v37 = v7;
  v40 = *(_QWORD *)(v5 + 96);
  v39 = *(_BYTE *)(v5 + 110);
  LOWORD(v7) = *(_WORD *)(v5 + 108);
  v43[0] = *(_QWORD *)(a1 + 88);
  v38 = v7;
  if ( v43[0] )
    sub_2AAAFA0(v43);
  sub_2BF1A90(a2, (__int64)v43);
  sub_9C6650(v43);
  v8 = sub_2BFB640(a2, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL), 0);
  if ( *(_BYTE *)(a1 + 161) )
  {
    v9 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * (unsigned int)(*(_DWORD *)(a1 + 56) - 1));
    if ( v9 )
    {
      v10 = 0;
      if ( !*(_BYTE *)(a2 + 12) )
        v10 = *(_DWORD *)(a2 + 8) == 1;
      v11 = sub_2BFB640(a2, v9, v10);
      v13 = *(_QWORD *)(v8 + 8);
      v14 = 0;
      v30 = v11;
      if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 <= 1 )
      {
        v14 = *(_QWORD *)(v8 + 8);
        v13 = *(_QWORD *)(v13 + 24);
      }
      v29 = v14;
      v15 = (__int64)sub_F70230(v34, v13, *(unsigned int *)(*(_QWORD *)(a1 + 152) + 44LL), v14, v12);
      if ( *(_BYTE *)(a2 + 12) )
      {
        if ( !*(_DWORD *)(a2 + 8) )
        {
LABEL_12:
          v17 = *(unsigned int ***)(a2 + 904);
          v44 = 257;
          v8 = sub_B36550(v17, v30, v8, v15, (__int64)v43, 0);
          goto LABEL_13;
        }
      }
      else if ( *(_DWORD *)(a2 + 8) <= 1u )
      {
        goto LABEL_12;
      }
      v16 = *(unsigned int ***)(a2 + 904);
      v44 = 257;
      LODWORD(v41) = *(_DWORD *)(v29 + 32);
      BYTE4(v41) = *(_BYTE *)(v29 + 8) == 18;
      v15 = sub_B37620(v16, v41, v15, v43);
      goto LABEL_12;
    }
  }
LABEL_13:
  if ( *(_BYTE *)(a1 + 160) )
  {
    v18 = *(_DWORD *)(a2 + 8);
    if ( *(_BYTE *)(a2 + 12) )
    {
      if ( v18 )
        goto LABEL_16;
    }
    else if ( v18 > 1 )
    {
LABEL_16:
      v19 = sub_F70430(*(_QWORD *)(a2 + 904), *(_QWORD *)(a1 + 152), v8, v32);
      goto LABEL_17;
    }
    v21 = *(_QWORD *)(a1 + 152);
    v22 = *(_QWORD *)(a2 + 904);
    v44 = 257;
    v35 = v22;
    v23 = sub_1022EF0(*(_DWORD *)(v21 + 40));
    v19 = sub_2C137C0(v35, v23, v32, v8, v42, (__int64)v43, 0);
    goto LABEL_17;
  }
  v33 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 1);
  v24 = sub_F703A0(*(_QWORD *)(a2 + 904), *(_QWORD *)(a1 + 152), v8, 0);
  if ( (unsigned int)(v34 - 6) > 3 )
  {
    v25 = *(_QWORD *)(a1 + 152);
    v26 = *(_QWORD *)(a2 + 904);
    v27 = *(_DWORD *)(v25 + 40);
    if ( (unsigned int)(v34 - 12) > 3 )
    {
      v31 = *(_QWORD *)(a2 + 904);
      v36 = v24;
      v44 = 257;
      v28 = sub_1022EF0(v27);
      v19 = sub_2C137C0(v31, v28, v36, v33, v42, (__int64)v43, 0);
      goto LABEL_17;
    }
  }
  else
  {
    v25 = *(_QWORD *)(a1 + 152);
    v26 = *(_QWORD *)(a2 + 904);
  }
  v19 = sub_F6F180(v26, *(_DWORD *)(v25 + 40), v24, v33);
LABEL_17:
  sub_2BF26E0(a2, a1 + 96, v19, 1);
  *(_QWORD *)(v5 + 96) = v40;
  *(_DWORD *)(v5 + 104) = v37;
  *(_WORD *)(v5 + 108) = v38;
  *(_BYTE *)(v5 + 110) = v39;
  return v39;
}
