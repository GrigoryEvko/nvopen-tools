// Function: sub_2EAE040
// Address: 0x2eae040
//
unsigned __int64 __fastcall sub_2EAE040(int *a1)
{
  __int64 v1; // rbx
  __int64 v2; // rbp
  char v3; // cl
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r13
  unsigned __int64 v11; // rsi
  __int64 v12; // r13
  _QWORD *v13; // r14
  unsigned __int64 i; // rax
  __int64 v15; // rax
  char v16; // dl
  int v17; // eax
  unsigned __int64 result; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  const char *v27; // rdi
  size_t v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  int v32; // eax
  int v33; // eax
  __int64 v34; // rax
  int v35; // eax
  int v36; // eax
  int v37; // eax
  size_t v38; // rax
  __int64 v39; // rdx
  int v40; // eax
  char v41; // al
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned int v44; // eax
  unsigned __int64 v45; // [rsp-60h] [rbp-60h]
  char v46; // [rsp-55h] [rbp-55h] BYREF
  unsigned int v47; // [rsp-54h] [rbp-54h] BYREF
  __int64 v48; // [rsp-50h] [rbp-50h] BYREF
  __int64 v49; // [rsp-48h] [rbp-48h] BYREF
  size_t v50; // [rsp-40h] [rbp-40h]
  __int64 v51; // [rsp-30h] [rbp-30h]
  __int64 v52; // [rsp-8h] [rbp-8h]

  v3 = *(_BYTE *)a1;
  v52 = v2;
  v51 = v1;
  switch ( v3 )
  {
    case 0:
      v41 = *((_BYTE *)a1 + 3);
      v46 = 0;
      LOBYTE(v47) = (v41 & 0x10) != 0;
      LODWORD(v49) = ((unsigned int)*a1 >> 8) & 0xFFF;
      LODWORD(v48) = a1[2];
      result = sub_2EAD680(&v46, (int *)&v48, (int *)&v49, &v47);
      break;
    case 1:
      v42 = *((_QWORD *)a1 + 3);
      LOBYTE(v47) = 1;
      v49 = v42;
      LODWORD(v48) = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EAD710((char *)&v47, (int *)&v48, &v49);
      break;
    case 2:
      v43 = *((_QWORD *)a1 + 3);
      LOBYTE(v47) = 2;
      v49 = v43;
      LODWORD(v48) = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EAD7A0((char *)&v47, (int *)&v48, &v49);
      break;
    case 3:
      v31 = *((_QWORD *)a1 + 3);
      LOBYTE(v47) = 3;
      v49 = v31;
      LODWORD(v48) = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EAD830((char *)&v47, (int *)&v48, &v49);
      break;
    case 4:
      v34 = *((_QWORD *)a1 + 3);
      LOBYTE(v47) = 4;
      v49 = v34;
      LODWORD(v48) = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EAD8C0((char *)&v47, (int *)&v48, &v49);
      break;
    case 5:
      v32 = a1[6];
      LOBYTE(v47) = 5;
      LODWORD(v49) = v32;
      LODWORD(v48) = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EAD950((char *)&v47, (int *)&v48, (int *)&v49);
      break;
    case 6:
    case 7:
      v19 = a1[8];
      v20 = (unsigned int)a1[2];
      v46 = v3;
      v49 = v20 | (v19 << 32);
      LODWORD(v48) = a1[6];
      v47 = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EAD9E0(&v46, (int *)&v47, (int *)&v48, &v49);
      break;
    case 8:
      v33 = a1[6];
      LOBYTE(v47) = 8;
      LODWORD(v49) = v33;
      LODWORD(v48) = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EAD950((char *)&v47, (int *)&v48, (int *)&v49);
      break;
    case 9:
      v27 = (const char *)*((_QWORD *)a1 + 3);
      v28 = 0;
      v49 = (__int64)v27;
      if ( v27 )
        v28 = strlen(v27);
      v50 = v28;
      v29 = a1[8];
      v30 = (unsigned int)a1[2];
      v46 = 9;
      v48 = v30 | (v29 << 32);
      v47 = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EADA70(&v46, (int *)&v47, &v48, (__int64)&v49);
      break;
    case 10:
      v23 = a1[8];
      v24 = (unsigned int)a1[2];
      v46 = 10;
      v49 = v24 | (v23 << 32);
      v48 = *((_QWORD *)a1 + 3);
      v47 = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EADB20(&v46, (int *)&v47, &v48, &v49);
      break;
    case 11:
      v25 = a1[8];
      v26 = (unsigned int)a1[2];
      v46 = 11;
      v49 = v26 | (v25 << 32);
      v48 = *((_QWORD *)a1 + 3);
      v47 = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EADBB0(&v46, (int *)&v47, &v48, &v49);
      break;
    case 12:
    case 13:
      v5 = *((_QWORD *)a1 + 2);
      if ( v5 && (v6 = *(_QWORD *)(v5 + 24)) != 0 && (v7 = *(_QWORD *)(v6 + 32)) != 0 )
      {
        v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v7 + 16) + 200LL))(*(_QWORD *)(v7 + 16));
        v9 = *((_QWORD *)a1 + 3);
        v10 = (unsigned int)(*(_DWORD *)(v8 + 16) + 31) >> 5;
        v11 = 8 * v10;
        v12 = 4 * v10;
        if ( v12 )
        {
          v13 = (_QWORD *)sub_22077B0(v11);
          for ( i = 0; i != v12; i += 4LL )
            v13[i / 4] = *(unsigned int *)(v9 + i);
        }
        else
        {
          v13 = 0;
          v11 = 0;
        }
        v15 = sub_CBF760(v13, v11);
        v16 = *(_BYTE *)a1;
        v49 = v15;
        v17 = 0;
        if ( v16 )
          v17 = ((unsigned int)*a1 >> 8) & 0xFFF;
        LOBYTE(v47) = v16;
        LODWORD(v48) = v17;
        result = sub_2EADC40((char *)&v47, (int *)&v48, &v49);
        if ( v13 )
        {
          v45 = result;
          j_j___libc_free_0((unsigned __int64)v13);
          result = v45;
        }
      }
      else
      {
        v44 = *a1;
        LOBYTE(v48) = v3;
        LODWORD(v49) = (v44 >> 8) & 0xFFF;
        result = sub_2EADCD0((char *)&v48, (int *)&v49);
      }
      break;
    case 14:
      v21 = *((_QWORD *)a1 + 3);
      LOBYTE(v47) = 14;
      v49 = v21;
      LODWORD(v48) = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EADD50((char *)&v47, (int *)&v48, &v49);
      break;
    case 15:
      v22 = *((_QWORD *)a1 + 3);
      LOBYTE(v47) = 15;
      v49 = v22;
      LODWORD(v48) = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EADDE0((char *)&v47, (int *)&v48, &v49);
      break;
    case 16:
      v35 = a1[6];
      LOBYTE(v47) = 16;
      LODWORD(v49) = v35;
      LODWORD(v48) = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EADF00((char *)&v47, (int *)&v48, (int *)&v49);
      break;
    case 17:
      v36 = a1[6];
      LOBYTE(v47) = 17;
      LODWORD(v49) = v36;
      LODWORD(v48) = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EADF00((char *)&v47, (int *)&v48, (int *)&v49);
      break;
    case 18:
      v37 = a1[6];
      LOBYTE(v47) = 18;
      LODWORD(v49) = v37;
      LODWORD(v48) = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EADF00((char *)&v47, (int *)&v48, (int *)&v49);
      break;
    case 19:
      v38 = *((_QWORD *)a1 + 4);
      v39 = *((_QWORD *)a1 + 3);
      LOBYTE(v47) = 19;
      v50 = v38;
      LODWORD(v38) = *a1;
      v49 = v39;
      LODWORD(v48) = ((unsigned int)v38 >> 8) & 0xFFF;
      result = sub_2EADF90((char *)&v47, (int *)&v48, (__int64)&v49);
      break;
    case 20:
      v40 = a1[7];
      v46 = 20;
      LODWORD(v49) = v40;
      LODWORD(v48) = a1[6];
      v47 = ((unsigned int)*a1 >> 8) & 0xFFF;
      result = sub_2EADE70(&v46, (int *)&v47, (int *)&v48, (int *)&v49);
      break;
    default:
      BUG();
  }
  return result;
}
