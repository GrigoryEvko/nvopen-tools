// Function: sub_2E8A880
// Address: 0x2e8a880
//
__int64 __fastcall sub_2E8A880(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 v4; // r12
  __int64 v6; // r13
  __int64 (*v7)(void); // rax
  int v8; // eax
  __int64 v9; // r13
  unsigned int v10; // r15d
  int v12; // eax
  int v13; // edx
  int v14; // eax
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rax
  int v18; // eax
  __int64 v19; // rax
  int v20; // eax
  __int64 v21; // rax
  __int64 (*v22)(); // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  int v25; // edx
  int v26; // edx
  unsigned int v27; // ecx
  __int64 (*v28)(); // rdx
  unsigned int v29; // eax
  char v30; // r14
  __int64 v31; // rdx
  __int64 *v32; // r12
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  __int64 **v35; // rbx
  unsigned int v36; // eax
  unsigned int v37; // eax
  int v38; // eax
  __int64 v39; // rax
  int v40; // eax
  __int64 v41; // rax
  __int64 **v42; // [rsp+10h] [rbp-50h]
  int v43; // [rsp+18h] [rbp-48h]
  __int64 **v44; // [rsp+18h] [rbp-48h]
  unsigned int v45; // [rsp+18h] [rbp-48h]
  __int64 v47; // [rsp+20h] [rbp-40h]
  __int64 **v48; // [rsp+28h] [rbp-38h]

  v4 = 0;
  v6 = sub_2E88D60(a1);
  v7 = *(__int64 (**)(void))(**(_QWORD **)(v6 + 16) + 128LL);
  if ( v7 != sub_2DAC790 )
    v4 = v7();
  v8 = *(_DWORD *)(a1 + 44);
  v9 = *(_QWORD *)(v6 + 48);
  if ( (v8 & 4) == 0 && (v8 & 8) != 0 )
  {
    if ( sub_2E88A90(a1, 128, 1) )
      return 1;
  }
  else if ( (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) & 0x80u) != 0LL )
  {
    return 1;
  }
  v12 = *(_DWORD *)(a3 + 44);
  if ( (v12 & 4) == 0 && (v12 & 8) != 0 )
  {
    LOBYTE(v37) = sub_2E88A90(a3, 128, 1);
    v10 = v37;
  }
  else
  {
    v10 = *(_QWORD *)(*(_QWORD *)(a3 + 16) + 24LL);
    LOBYTE(v10) = (unsigned __int8)v10 >> 7;
  }
  if ( (_BYTE)v10 )
    return 1;
  if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 > 1
    || (LOBYTE(v13) = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 64LL), (v13 & 0x10) == 0) )
  {
    v14 = *(_DWORD *)(a1 + 44);
    if ( (v14 & 4) == 0 && (v14 & 8) != 0 )
      LOBYTE(v15) = sub_2E88A90(a1, 0x100000, 1);
    else
      v15 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 20) & 1LL;
    if ( !(_BYTE)v15
      && ((unsigned int)*(unsigned __int16 *)(a3 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(a3 + 32) + 64LL) & 0x10) == 0) )
    {
      v16 = *(_DWORD *)(a3 + 44);
      if ( (v16 & 4) != 0 || (v16 & 8) == 0 )
        v17 = (*(_QWORD *)(*(_QWORD *)(a3 + 16) + 24LL) >> 20) & 1LL;
      else
        LOBYTE(v17) = sub_2E88A90(a3, 0x100000, 1);
      if ( !(_BYTE)v17 )
        return v10;
    }
    if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 > 1 )
      goto LABEL_25;
    v13 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 64LL);
  }
  if ( (v13 & 8) == 0 )
  {
LABEL_25:
    v18 = *(_DWORD *)(a1 + 44);
    if ( (v18 & 4) == 0 && (v18 & 8) != 0 )
      LOBYTE(v19) = sub_2E88A90(a1, 0x80000, 1);
    else
      v19 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 19) & 1LL;
    if ( !(_BYTE)v19
      && ((unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 0x10) == 0) )
    {
      v38 = *(_DWORD *)(a1 + 44);
      if ( (v38 & 4) != 0 || (v38 & 8) == 0 )
        v39 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 20) & 1LL;
      else
        LOBYTE(v39) = sub_2E88A90(a1, 0x100000, 1);
      if ( !(_BYTE)v39 )
        return v10;
    }
  }
  if ( (unsigned int)*(unsigned __int16 *)(a3 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(a3 + 32) + 64LL) & 8) != 0
    || ((v20 = *(_DWORD *)(a3 + 44), (v20 & 4) != 0) || (v20 & 8) == 0
      ? (v21 = (*(_QWORD *)(*(_QWORD *)(a3 + 16) + 24LL) >> 19) & 1LL)
      : (LOBYTE(v21) = sub_2E88A90(a3, 0x80000, 1)),
        (_BYTE)v21
     || (unsigned int)*(unsigned __int16 *)(a3 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(a3 + 32) + 64LL) & 0x10) != 0
     || ((v40 = *(_DWORD *)(a3 + 44), (v40 & 4) != 0) || (v40 & 8) == 0
       ? (v41 = (*(_QWORD *)(*(_QWORD *)(a3 + 16) + 24LL) >> 20) & 1LL)
       : (LOBYTE(v41) = sub_2E88A90(a3, 0x100000, 1)),
         (_BYTE)v41)) )
  {
    v22 = *(__int64 (**)())(*(_QWORD *)v4 + 1264LL);
    if ( v22 == sub_2E85460 || !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64))v22)(v4, a1, a3) )
    {
      sub_2E864A0(a1);
      if ( !v23 )
        return 1;
      sub_2E864A0(a3);
      if ( !v24 )
        return 1;
      sub_2E864A0(a1);
      v43 = v25;
      sub_2E864A0(a3);
      v27 = v26 * v43;
      v28 = *(__int64 (**)())(*(_QWORD *)v4 + 1280LL);
      v29 = 16;
      if ( v28 != sub_2E85470 )
      {
        v45 = v27;
        v29 = ((__int64 (__fastcall *)(__int64))v28)(v4);
        v27 = v45;
      }
      if ( v29 < v27 )
        return 1;
      v30 = a4;
      v44 = (__int64 **)sub_2E864A0(a1);
      v42 = &v44[v31];
      if ( v44 != v42 )
      {
        v47 = a3;
        while ( 1 )
        {
          v32 = *v44;
          v33 = sub_2E864A0(v47);
          v48 = (__int64 **)(v33 + 8 * v34);
          v35 = (__int64 **)v33;
          if ( (__int64 **)v33 != v48 )
            break;
LABEL_75:
          if ( v42 == ++v44 )
            return 0;
        }
        while ( 1 )
        {
          LOBYTE(v36) = sub_2E85750(v9, a2, v30, v32, *v35);
          if ( (_BYTE)v36 )
            return v36;
          if ( v48 == ++v35 )
            goto LABEL_75;
        }
      }
    }
  }
  return v10;
}
