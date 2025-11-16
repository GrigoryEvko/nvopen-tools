// Function: sub_31C08E0
// Address: 0x31c08e0
//
__int64 __fastcall sub_31C08E0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rbx
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned int v15; // ecx
  __int64 *v16; // rdx
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rcx
  unsigned int v22; // esi
  __int64 *v23; // rdx
  __int64 v24; // r8
  __int64 *v25; // r12
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rdi
  __int64 result; // rax
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // rcx
  unsigned int v33; // edi
  __int64 *v34; // rdx
  __int64 v35; // r8
  __int64 v36; // rax
  _BYTE *v37; // rsi
  _BYTE *v38; // rsi
  int v39; // edx
  int v40; // r9d
  int v41; // edx
  int v42; // r9d
  __int64 v43; // rdx
  int v44; // edx
  int v45; // r9d
  __int64 v46; // [rsp+0h] [rbp-1C0h]
  __int64 v47; // [rsp+8h] [rbp-1B8h]
  __int64 v48; // [rsp+8h] [rbp-1B8h]
  _QWORD v49[12]; // [rsp+10h] [rbp-1B0h] BYREF
  _QWORD v50[12]; // [rsp+70h] [rbp-150h] BYREF
  _QWORD v51[12]; // [rsp+D0h] [rbp-F0h] BYREF
  _QWORD v52[18]; // [rsp+130h] [rbp-90h] BYREF

  if ( !*(_BYTE *)(a1 + 208) )
    abort();
  v5 = sub_371B3B0(a1 + 176, *(_QWORD *)(a1 + 184), *(_QWORD *)(a1 + 192));
  v6 = *a2;
  v7 = sub_31BFEB0((__int64)a2, a3, 1);
  v47 = v8;
  v9 = v7;
  if ( v7 != v8 )
  {
    do
    {
      v10 = *(_QWORD *)v9;
      if ( sub_B445A0(*(_QWORD *)(v6 + 16), *(_QWORD *)(*(_QWORD *)v9 + 16LL)) )
        v6 = v10;
      v9 += 8;
    }
    while ( v47 != v9 );
    v11 = sub_318B520(v5);
    if ( v6 != v11 )
      goto LABEL_7;
LABEL_40:
    v48 = v6;
    v6 = sub_318B4B0(v6);
    goto LABEL_16;
  }
  v11 = sub_318B520(v5);
  if ( v11 != v6 )
  {
LABEL_7:
    v12 = v6;
    while ( 1 )
    {
      v13 = *(unsigned int *)(a1 + 64);
      v14 = *(_QWORD *)(a1 + 48);
      if ( !(_DWORD)v13 )
        goto LABEL_8;
      v15 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( *v16 != v12 )
      {
        v44 = 1;
        while ( v17 != -4096 )
        {
          v45 = v44 + 1;
          v15 = (v13 - 1) & (v44 + v15);
          v16 = (__int64 *)(v14 + 16LL * v15);
          v17 = *v16;
          if ( *v16 == v12 )
            goto LABEL_11;
          v44 = v45;
        }
        goto LABEL_8;
      }
LABEL_11:
      if ( v16 != (__int64 *)(v14 + 16 * v13)
        && (v18 = v16[1]) != 0
        && (v19 = *(_QWORD *)(v18 + 32), *(_DWORD *)(v19 + 8) == 1) )
      {
        sub_31C00E0(a1, v19);
        v12 = sub_318B520(v12);
        if ( v12 == v11 )
          break;
      }
      else
      {
LABEL_8:
        v12 = sub_318B520(v12);
        if ( v12 == v11 )
          break;
      }
    }
  }
  v48 = 0;
  if ( v6 )
    goto LABEL_40;
LABEL_16:
  v46 = a1 + 40;
  if ( v5 != v6 )
  {
    while ( 1 )
    {
      v20 = *(unsigned int *)(a1 + 64);
      v21 = *(_QWORD *)(a1 + 48);
      if ( !(_DWORD)v20 )
        goto LABEL_60;
      v22 = (v20 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v23 = (__int64 *)(v21 + 16LL * v22);
      v24 = *v23;
      if ( v5 != *v23 )
        break;
LABEL_19:
      if ( v23 == (__int64 *)(v21 + 16 * v20) )
        goto LABEL_60;
      v25 = (__int64 *)v23[1];
      v26 = *v25;
      *((_BYTE *)v25 + 24) = 0;
      *((_DWORD *)v25 + 5) = 0;
      (*(void (__fastcall **)(_QWORD *, __int64 *, __int64))(v26 + 24))(v52, v25, v46);
      (*(void (__fastcall **)(_QWORD *, __int64 *, __int64))(*v25 + 16))(v51, v25, v46);
      v49[0] = v51[0];
      v49[1] = v51[1];
      v49[2] = v51[2];
      v49[3] = v51[3];
      v49[4] = v51[4];
      v49[5] = v51[5];
      v49[6] = v51[6];
      v49[7] = v51[7];
      v49[8] = v51[8];
      v49[9] = v51[9];
      v49[10] = v51[10];
      v49[11] = v51[11];
      v50[0] = v52[0];
      v50[1] = v52[1];
      v50[2] = v52[2];
      v50[3] = v52[3];
      v50[4] = v52[4];
      v50[5] = v52[5];
      v50[6] = v52[6];
      v50[7] = v52[7];
      v50[8] = v52[8];
      v50[9] = v52[9];
      v50[10] = v52[10];
      v50[11] = v52[11];
      while ( !sub_31B8DE0(v49, v50) )
      {
        v27 = sub_31B8B80((__int64)v49);
        ++*(_DWORD *)(v27 + 20);
        sub_31B8D10((__int64)v49);
      }
      v5 = sub_318B4B0(v5);
      if ( v5 == v6 )
        goto LABEL_24;
    }
    v39 = 1;
    while ( v24 != -4096 )
    {
      v40 = v39 + 1;
      v22 = (v20 - 1) & (v22 + v39);
      v23 = (__int64 *)(v21 + 16LL * v22);
      v24 = *v23;
      if ( *v23 == v5 )
        goto LABEL_19;
      v39 = v40;
    }
LABEL_60:
    MEMORY[0x14] = 0;
    BUG();
  }
LABEL_24:
  v28 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  if ( v28 )
    j_j___libc_free_0(v28);
  result = v48;
  v30 = *(_QWORD *)(a1 + 72);
  if ( v48 )
  {
    result = sub_318B4B0(v48);
    v48 = result;
  }
  if ( v30 != v48 )
  {
    while ( 1 )
    {
      v31 = *(unsigned int *)(a1 + 64);
      v32 = *(_QWORD *)(a1 + 48);
      if ( !(_DWORD)v31 )
        goto LABEL_59;
      v33 = (v31 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v34 = (__int64 *)(v32 + 16LL * v33);
      v35 = *v34;
      if ( v30 != *v34 )
        break;
LABEL_33:
      if ( v34 == (__int64 *)(v32 + 16 * v31) )
        goto LABEL_59;
      v36 = v34[1];
      if ( !*(_DWORD *)(v36 + 20) )
      {
        v50[0] = v34[1];
        v37 = *(_BYTE **)(a1 + 16);
        if ( v37 == *(_BYTE **)(a1 + 24) )
        {
          sub_31C0410(a1 + 8, v37, v50);
          v38 = *(_BYTE **)(a1 + 16);
        }
        else
        {
          if ( v37 )
          {
            *(_QWORD *)v37 = v36;
            v37 = *(_BYTE **)(a1 + 16);
          }
          v38 = v37 + 8;
          *(_QWORD *)(a1 + 16) = v38;
        }
        sub_31BFEC0(*(_QWORD *)(a1 + 8), ((__int64)&v38[-*(_QWORD *)(a1 + 8)] >> 3) - 1, 0, *((_QWORD *)v38 - 1));
      }
      result = sub_318B4B0(v30);
      v30 = result;
      if ( result == v48 )
        return result;
    }
    v41 = 1;
    while ( v35 != -4096 )
    {
      v42 = v41 + 1;
      v43 = ((_DWORD)v31 - 1) & (v33 + v41);
      v33 = v43;
      v34 = (__int64 *)(v32 + 16 * v43);
      v35 = *v34;
      if ( v30 == *v34 )
        goto LABEL_33;
      v41 = v42;
    }
LABEL_59:
    BUG();
  }
  return result;
}
