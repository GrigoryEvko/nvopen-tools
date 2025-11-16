// Function: sub_22BEEC0
// Address: 0x22beec0
//
__int64 __fastcall sub_22BEEC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 result; // rax
  int v5; // edx
  unsigned int v6; // eax
  _QWORD *v7; // r12
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  bool v12; // zf
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // r15
  __int64 v16; // rax
  bool v17; // r12
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // r8
  int v21; // r9d
  unsigned int v22; // ecx
  __int64 v23; // rax
  __int64 v24; // r10
  __int64 v25; // rcx
  unsigned int v26; // eax
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // r8
  int v30; // r9d
  unsigned int v31; // esi
  _QWORD *v32; // rdi
  __int64 v33; // r10
  __int64 v34; // rax
  unsigned int v35; // eax
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // esi
  int v39; // eax
  __int64 v40; // rcx
  __int64 v41; // r9
  int v42; // r8d
  unsigned int v43; // esi
  _QWORD *v44; // rdi
  __int64 v45; // r10
  __int64 v46; // rcx
  unsigned int v47; // ecx
  int v48; // esi
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // rdi
  int v51; // eax
  int v52; // edx
  int v53; // edi
  int v54; // edx
  int v55; // edi
  int v56; // edx
  __int64 v57; // rdi
  int v58; // edi
  __int64 v59; // [rsp+0h] [rbp-B0h]
  __int64 v60; // [rsp+0h] [rbp-B0h]
  __int64 v61; // [rsp+0h] [rbp-B0h]
  __int64 v62; // [rsp+0h] [rbp-B0h]
  __int64 v64; // [rsp+10h] [rbp-A0h]
  __int64 v65; // [rsp+10h] [rbp-A0h]
  __int64 v66; // [rsp+10h] [rbp-A0h]
  __int64 v67; // [rsp+10h] [rbp-A0h]
  __int64 v68; // [rsp+10h] [rbp-A0h]
  __int64 v69; // [rsp+18h] [rbp-98h]
  void *v70; // [rsp+20h] [rbp-90h] BYREF
  __int64 v71; // [rsp+28h] [rbp-88h] BYREF
  __int64 v72; // [rsp+30h] [rbp-80h]
  __int64 v73; // [rsp+38h] [rbp-78h]
  char v74; // [rsp+40h] [rbp-70h]
  __int64 (__fastcall **v75)(); // [rsp+50h] [rbp-60h] BYREF
  __int64 v76; // [rsp+58h] [rbp-58h] BYREF
  __int64 v77; // [rsp+60h] [rbp-50h]
  __int64 i; // [rsp+68h] [rbp-48h]
  __int64 v79; // [rsp+70h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 16) )
  {
    v71 = 2;
    v72 = 0;
    v13 = *(unsigned int *)(a1 + 24);
    v14 = *(_QWORD *)(a1 + 8);
    v74 = 0;
    v70 = &unk_49DE8C0;
    v73 = -4096;
    v13 *= 48;
    v76 = 2;
    v15 = v14 + v13;
    v77 = 0;
    LOBYTE(v79) = 0;
    for ( i = -8192; v15 != v14; v14 += 48 )
    {
      v16 = *(_QWORD *)(v14 + 24);
      if ( v16 != -8192 && v16 != -4096 )
        break;
    }
    v75 = (__int64 (__fastcall **)())&unk_49DB368;
    sub_D68D70(&v76);
    v70 = &unk_49DB368;
    sub_D68D70(&v71);
    v69 = *(_QWORD *)(a1 + 8) + 48LL * *(unsigned int *)(a1 + 24);
    if ( v69 != v14 )
    {
      v17 = a2 != -4096 && a2 != 0 && a2 != -8192;
      do
      {
        v18 = *(_QWORD *)(v14 + 40);
        v19 = a2;
        v72 = a2;
        v70 = 0;
        v71 = 0;
        if ( v17 )
        {
          sub_BD73F0((__int64)&v70);
          v19 = v72;
        }
        if ( (*(_BYTE *)(v18 + 8) & 1) != 0 )
        {
          v20 = v18 + 16;
          v21 = 3;
        }
        else
        {
          v39 = *(_DWORD *)(v18 + 24);
          v20 = *(_QWORD *)(v18 + 16);
          v21 = v39 - 1;
          if ( !v39 )
            goto LABEL_34;
        }
        v22 = v21 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v23 = v20 + ((unsigned __int64)v22 << 6);
        v24 = *(_QWORD *)(v23 + 16);
        if ( v24 == v19 )
        {
LABEL_25:
          if ( (unsigned int)*(unsigned __int8 *)(v23 + 24) - 4 <= 1 )
          {
            if ( *(_DWORD *)(v23 + 56) > 0x40u )
            {
              v49 = *(_QWORD *)(v23 + 48);
              if ( v49 )
              {
                v61 = v23;
                j_j___libc_free_0_0(v49);
                v23 = v61;
              }
            }
            if ( *(_DWORD *)(v23 + 40) > 0x40u )
            {
              v50 = *(_QWORD *)(v23 + 32);
              if ( v50 )
              {
                v62 = v23;
                j_j___libc_free_0_0(v50);
                v23 = v62;
              }
            }
          }
          v75 = 0;
          v76 = 0;
          v77 = -8192;
          v25 = *(_QWORD *)(v23 + 16);
          if ( v25 != -8192 )
          {
            if ( v25 != -4096 && v25 )
            {
              v64 = v23;
              sub_BD60C0((_QWORD *)v23);
              v23 = v64;
            }
            *(_QWORD *)(v23 + 16) = -8192;
            if ( v77 != -4096 && v77 != 0 && v77 != -8192 )
              sub_BD60C0(&v75);
          }
          v26 = *(_DWORD *)(v18 + 8);
          ++*(_DWORD *)(v18 + 12);
          *(_DWORD *)(v18 + 8) = (2 * (v26 >> 1) - 2) | v26 & 1;
          v19 = v72;
        }
        else
        {
          v51 = 1;
          while ( v24 != -4096 )
          {
            v52 = v51 + 1;
            v22 = v21 & (v51 + v22);
            v23 = v20 + ((unsigned __int64)v22 << 6);
            v24 = *(_QWORD *)(v23 + 16);
            if ( v24 == v19 )
              goto LABEL_25;
            v51 = v52;
          }
        }
LABEL_34:
        if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
          sub_BD60C0(&v70);
        v27 = *(_QWORD *)(v14 + 40);
        v28 = a2;
        v72 = a2;
        v70 = 0;
        v71 = 0;
        if ( v17 )
        {
          v65 = v27;
          sub_BD73F0((__int64)&v70);
          v28 = v72;
          v27 = v65;
        }
        if ( (*(_BYTE *)(v27 + 280) & 1) != 0 )
        {
          v29 = v27 + 288;
          v30 = 3;
        }
        else
        {
          v38 = *(_DWORD *)(v27 + 296);
          v29 = *(_QWORD *)(v27 + 288);
          v30 = v38 - 1;
          if ( !v38 )
            goto LABEL_50;
        }
        v31 = v30 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v32 = (_QWORD *)(v29 + 24LL * v31);
        v33 = v32[2];
        if ( v28 == v33 )
        {
LABEL_42:
          v75 = 0;
          v76 = 0;
          v77 = -8192;
          v34 = v32[2];
          if ( v34 != -8192 )
          {
            if ( v34 && v34 != -4096 )
            {
              v59 = v27;
              sub_BD60C0(v32);
              v27 = v59;
            }
            v32[2] = -8192;
            if ( v77 != 0 && v77 != -4096 && v77 != -8192 )
            {
              v66 = v27;
              sub_BD60C0(&v75);
              v27 = v66;
            }
          }
          v35 = *(_DWORD *)(v27 + 280);
          ++*(_DWORD *)(v27 + 284);
          *(_DWORD *)(v27 + 280) = (2 * (v35 >> 1) - 2) | v35 & 1;
          v28 = v72;
        }
        else
        {
          v53 = 1;
          while ( v33 != -4096 )
          {
            v54 = v53 + 1;
            v31 = v30 & (v53 + v31);
            v32 = (_QWORD *)(v29 + 24LL * v31);
            v33 = v32[2];
            if ( v33 == v28 )
              goto LABEL_42;
            v53 = v54;
          }
        }
LABEL_50:
        if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
          sub_BD60C0(&v70);
        v36 = *(_QWORD *)(v14 + 40);
        if ( !*(_BYTE *)(v36 + 448) )
          goto LABEL_54;
        v70 = 0;
        v40 = a2;
        v71 = 0;
        v72 = a2;
        if ( v17 )
        {
          v67 = v36;
          sub_BD73F0((__int64)&v70);
          v40 = v72;
          v36 = v67;
        }
        if ( (*(_BYTE *)(v36 + 392) & 1) != 0 )
        {
          v41 = v36 + 400;
          v42 = 1;
LABEL_68:
          v43 = v42 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
          v44 = (_QWORD *)(v41 + 24LL * v43);
          v45 = v44[2];
          if ( v40 == v45 )
          {
LABEL_69:
            v75 = 0;
            v76 = 0;
            v77 = -8192;
            v46 = v44[2];
            if ( v46 != -8192 )
            {
              if ( v46 != -4096 && v46 )
              {
                v60 = v36;
                sub_BD60C0(v44);
                v36 = v60;
              }
              v44[2] = -8192;
              if ( v77 != -4096 && v77 != 0 && v77 != -8192 )
              {
                v68 = v36;
                sub_BD60C0(&v75);
                v36 = v68;
              }
            }
            v47 = *(_DWORD *)(v36 + 392);
            ++*(_DWORD *)(v36 + 396);
            *(_DWORD *)(v36 + 392) = (2 * (v47 >> 1) - 2) | v47 & 1;
            v40 = v72;
          }
          else
          {
            v55 = 1;
            while ( v45 != -4096 )
            {
              v56 = v55 + 1;
              v57 = v42 & (v43 + v55);
              v43 = v57;
              v44 = (_QWORD *)(v41 + 24 * v57);
              v45 = v44[2];
              if ( v45 == v40 )
                goto LABEL_69;
              v55 = v56;
            }
          }
          goto LABEL_77;
        }
        v48 = *(_DWORD *)(v36 + 408);
        v41 = *(_QWORD *)(v36 + 400);
        v42 = v48 - 1;
        if ( v48 )
          goto LABEL_68;
LABEL_77:
        if ( v40 != 0 && v40 != -4096 && v40 != -8192 )
          sub_BD60C0(&v70);
LABEL_54:
        for ( v14 += 48; v15 != v14; v14 += 48 )
        {
          v37 = *(_QWORD *)(v14 + 24);
          if ( v37 != -4096 && v37 != -8192 )
            break;
        }
      }
      while ( v69 != v14 );
    }
  }
  v3 = *(_QWORD *)(a1 + 40);
  result = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)result )
  {
    v5 = result - 1;
    v76 = 2;
    v77 = 0;
    i = -4096;
    v6 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v79 = 0;
    v7 = (_QWORD *)(v3 + 40LL * v6);
    v8 = v7[3];
    if ( v8 == a2 )
    {
LABEL_4:
      v75 = (__int64 (__fastcall **)())&unk_49DB368;
      sub_D68D70(&v76);
      result = *(_QWORD *)(a1 + 40) + 40LL * *(unsigned int *)(a1 + 56);
      if ( v7 != (_QWORD *)result )
      {
        v76 = 2;
        v9 = 0;
        v77 = 0;
        i = -8192;
        v75 = off_4A09D90;
        v79 = 0;
        v10 = v7[3];
        if ( v10 != -8192 )
        {
          if ( !v10 || v10 == -4096 )
          {
            v7[3] = -8192;
            v9 = v79;
          }
          else
          {
            sub_BD60C0(v7 + 1);
            v11 = i;
            v12 = i == 0;
            v7[3] = i;
            if ( v11 != -4096 && !v12 && v11 != -8192 )
              sub_BD6050(v7 + 1, v76 & 0xFFFFFFFFFFFFFFF8LL);
            v9 = v79;
          }
        }
        v7[4] = v9;
        v75 = (__int64 (__fastcall **)())&unk_49DB368;
        sub_D68D70(&v76);
        result = a1;
        --*(_DWORD *)(a1 + 48);
        ++*(_DWORD *)(a1 + 52);
      }
    }
    else
    {
      v58 = 1;
      while ( v8 != -4096 )
      {
        v6 = v5 & (v58 + v6);
        v7 = (_QWORD *)(v3 + 40LL * v6);
        v8 = v7[3];
        if ( v8 == a2 )
          goto LABEL_4;
        ++v58;
      }
      v75 = (__int64 (__fastcall **)())&unk_49DB368;
      return sub_D68D70(&v76);
    }
  }
  return result;
}
