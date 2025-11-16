// Function: sub_B02F70
// Address: 0xb02f70
//
__int64 __fastcall sub_B02F70(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6, char a7)
{
  __int64 v7; // r10
  __int64 *v9; // r14
  unsigned int v11; // r12d
  __int64 v12; // r9
  int v13; // ebx
  __int64 v14; // rax
  unsigned int v15; // edx
  __int64 *v16; // rsi
  __int64 v17; // rax
  int v18; // eax
  __int64 v19; // r9
  __int64 v20; // r10
  __int64 v21; // r8
  unsigned int v22; // ebx
  int v23; // r12d
  __int64 v24; // r14
  _BYTE *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  unsigned int v29; // edx
  __int64 *v30; // rdi
  __int64 v31; // rsi
  unsigned int v32; // edx
  __int64 *v33; // rdi
  __int64 v34; // rax
  int v35; // eax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 *v39; // rdi
  unsigned int v40; // eax
  __int64 v41; // rsi
  __int64 *v42; // rdi
  unsigned int v43; // edx
  __int64 v44; // rax
  _BYTE *v45; // rax
  _BYTE *v46; // rax
  __int64 v47; // rcx
  __int64 v48; // rbx
  __int64 result; // rax
  __int64 v50; // rax
  __int64 v51; // r15
  __int64 v52; // rax
  __int64 v53; // r13
  int v54; // [rsp+Ch] [rbp-A4h]
  __int64 v55; // [rsp+10h] [rbp-A0h]
  __int64 v56; // [rsp+18h] [rbp-98h]
  __int64 v57; // [rsp+20h] [rbp-90h]
  __int64 v58; // [rsp+28h] [rbp-88h]
  __int64 v59; // [rsp+28h] [rbp-88h]
  __int64 v62; // [rsp+38h] [rbp-78h]
  __int64 v64; // [rsp+38h] [rbp-78h]
  __int64 v65; // [rsp+40h] [rbp-70h]
  __int64 v66; // [rsp+58h] [rbp-58h] BYREF
  __int64 v67; // [rsp+60h] [rbp-50h] BYREF
  __int64 v68; // [rsp+68h] [rbp-48h] BYREF
  __int64 v69; // [rsp+70h] [rbp-40h] BYREF
  __int64 v70[7]; // [rsp+78h] [rbp-38h] BYREF

  v7 = a4;
  v9 = a1;
  v11 = a6;
  if ( a6 )
  {
LABEL_41:
    v50 = *v9;
    v67 = a2;
    v68 = a3;
    v51 = v50 + 824;
    v69 = v7;
    v70[0] = a5;
    v52 = sub_B97910(16, 4, v11);
    v53 = v52;
    if ( v52 )
      sub_AF2740(v52, (int)v9, v11, (int)&v67, 4);
    return sub_B02E90(v53, v11, v51);
  }
  v12 = *a1;
  v67 = a2;
  v68 = a3;
  v69 = a4;
  v70[0] = a5;
  v13 = *(_DWORD *)(v12 + 848);
  v65 = *(_QWORD *)(v12 + 832);
  if ( !v13 )
    goto LABEL_40;
  if ( a2 && *(_BYTE *)a2 == 1 )
  {
    v14 = *(_QWORD *)(a2 + 136);
    v15 = *(_DWORD *)(v14 + 32);
    v16 = *(__int64 **)(v14 + 24);
    if ( v15 > 0x40 )
    {
      v17 = *v16;
    }
    else
    {
      v17 = 0;
      if ( v15 )
        v17 = (__int64)((_QWORD)v16 << (64 - (unsigned __int8)v15)) >> (64 - (unsigned __int8)v15);
    }
    v58 = a5;
    v62 = v12;
    v66 = v17;
    v18 = sub_AF7D50(&v66, &v68, &v69, v70);
    v19 = v62;
    v20 = a4;
    v21 = v58;
  }
  else
  {
    v59 = a5;
    v64 = v12;
    v18 = sub_AF81D0(&v67, &v68, &v69, v70);
    v21 = v59;
    v20 = a4;
    v19 = v64;
  }
  v55 = v19;
  v57 = v20;
  v56 = v21;
  v54 = v13 - 1;
  v22 = (v13 - 1) & v18;
  v23 = 1;
  while ( 1 )
  {
    v24 = *(_QWORD *)(v65 + 8LL * v22);
    if ( v24 == -4096 )
    {
      v9 = a1;
      v7 = v57;
      a5 = v56;
      v11 = 0;
      goto LABEL_40;
    }
    if ( v24 != -8192 )
    {
      v25 = sub_A17150((_BYTE *)(v24 - 16));
      v26 = *(_QWORD *)v25;
      if ( *(_QWORD *)v25 == v67 )
        goto LABEL_24;
      if ( v67 && *(_BYTE *)v67 == 1 && v26 && *(_BYTE *)v26 == 1 )
      {
        v27 = *(_QWORD *)(v67 + 136);
        v28 = *(_QWORD *)(v26 + 136);
        v29 = *(_DWORD *)(v27 + 32);
        v30 = *(__int64 **)(v27 + 24);
        if ( v29 <= 0x40 )
        {
          v31 = 0;
          if ( v29 )
            v31 = (__int64)((_QWORD)v30 << (64 - (unsigned __int8)v29)) >> (64 - (unsigned __int8)v29);
        }
        else
        {
          v31 = *v30;
        }
        v32 = *(_DWORD *)(v28 + 32);
        v33 = *(__int64 **)(v28 + 24);
        if ( v32 > 0x40 )
        {
          v34 = *v33;
        }
        else
        {
          v34 = 0;
          if ( v32 )
            v34 = (__int64)((_QWORD)v33 << (64 - (unsigned __int8)v32)) >> (64 - (unsigned __int8)v32);
        }
        if ( v34 == v31 )
        {
LABEL_24:
          v36 = *((_QWORD *)sub_A17150((_BYTE *)(v24 - 16)) + 1);
          if ( v36 == v68 )
            goto LABEL_55;
          if ( v68 && *(_BYTE *)v68 == 1 && v36 && *(_BYTE *)v36 == 1 )
          {
            v37 = *(_QWORD *)(v68 + 136);
            v38 = *(_QWORD *)(v36 + 136);
            v39 = *(__int64 **)(v37 + 24);
            v40 = *(_DWORD *)(v37 + 32);
            if ( v40 > 0x40 )
            {
              v41 = *v39;
            }
            else
            {
              v41 = 0;
              if ( v40 )
                v41 = (__int64)((_QWORD)v39 << (64 - (unsigned __int8)v40)) >> (64 - (unsigned __int8)v40);
            }
            v42 = *(__int64 **)(v38 + 24);
            v43 = *(_DWORD *)(v38 + 32);
            if ( v43 > 0x40 )
            {
              v44 = *v42;
            }
            else
            {
              v44 = 0;
              if ( v43 )
                v44 = (__int64)((_QWORD)v42 << (64 - (unsigned __int8)v43)) >> (64 - (unsigned __int8)v43);
            }
            if ( v44 == v41 )
            {
LABEL_55:
              v45 = sub_A17150((_BYTE *)(v24 - 16));
              if ( sub_AF1330(v69, *((_QWORD *)v45 + 2)) )
              {
                v46 = sub_A17150((_BYTE *)(v24 - 16));
                if ( sub_AF1330(v70[0], *((_QWORD *)v46 + 3)) )
                  break;
              }
            }
          }
        }
      }
    }
    v35 = v54 & (v22 + v23++);
    v22 = v35;
  }
  v47 = v65 + 8LL * v22;
  v48 = v24;
  v7 = v57;
  a5 = v56;
  v11 = 0;
  v9 = a1;
  if ( v47 == *(_QWORD *)(v55 + 832) + 8LL * *(unsigned int *)(v55 + 848) || (result = v48) == 0 )
  {
LABEL_40:
    result = 0;
    if ( !a7 )
      return result;
    goto LABEL_41;
  }
  return result;
}
