// Function: sub_22B2B70
// Address: 0x22b2b70
//
unsigned __int64 __fastcall sub_22B2B70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // r15
  __int64 v10; // rax
  _BYTE *v11; // rdx
  __int64 *v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // eax
  unsigned __int64 v17; // rax
  unsigned __int8 *v18; // rbx
  unsigned __int64 v19; // r12
  char *v21; // rax
  unsigned __int8 *v22; // rbx
  __int64 v23; // rax
  _BYTE *v24; // rbx
  size_t v25; // r12
  char *v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int8 *v28; // rbx
  unsigned __int64 v29; // r12
  unsigned int v30; // ebx
  char *v31; // rax
  char *v32; // rdi
  __int64 v33; // rsi
  unsigned __int64 v34; // rax
  unsigned __int8 *v35; // rbx
  char *v36; // rdi
  unsigned __int64 v37; // [rsp+18h] [rbp-A8h] BYREF
  unsigned __int64 v38; // [rsp+20h] [rbp-A0h] BYREF
  unsigned __int64 v39; // [rsp+28h] [rbp-98h] BYREF
  unsigned __int64 v40; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v41; // [rsp+38h] [rbp-88h] BYREF
  char *v42; // [rsp+40h] [rbp-80h] BYREF
  size_t v43; // [rsp+48h] [rbp-78h]
  _QWORD v44[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD *v45; // [rsp+60h] [rbp-60h] BYREF
  __int64 v46; // [rsp+68h] [rbp-58h]
  _BYTE v47[80]; // [rsp+70h] [rbp-50h] BYREF

  v46 = 0x400000000LL;
  v7 = *(__int64 **)(a1 + 24);
  v8 = *(unsigned int *)(a1 + 32);
  v45 = v47;
  v9 = &v7[v8];
  if ( v7 != v9 )
  {
    v10 = *v7;
    v11 = v47;
    v12 = v7 + 1;
    v13 = *(_QWORD *)(v10 + 8);
    v14 = 0;
    while ( 1 )
    {
      *(_QWORD *)&v11[8 * v14] = v13;
      v14 = (unsigned int)(v46 + 1);
      LODWORD(v46) = v46 + 1;
      if ( v9 == v12 )
        break;
      v13 = *(_QWORD *)(*v12 + 8);
      if ( v14 + 1 > (unsigned __int64)HIDWORD(v46) )
      {
        sub_C8D5F0((__int64)&v45, v47, v14 + 1, 8u, a5, a6);
        v14 = (unsigned int)v46;
      }
      v11 = v45;
      ++v12;
    }
  }
  v15 = *(_QWORD *)(a1 + 16);
  if ( (unsigned __int8)(*(_BYTE *)v15 - 82) <= 1u )
  {
    v42 = (char *)sub_22B0AE0(v45, (__int64)&v45[(unsigned int)v46]);
    v16 = sub_22AF4B0(a1);
    v17 = sub_22AE640(v16);
    v18 = *(unsigned __int8 **)(a1 + 16);
    v41 = v17;
    v40 = sub_22AE640(*((_QWORD *)v18 + 1));
    v39 = sub_22AE640((unsigned int)*v18 - 29);
    v19 = sub_22B2710((__int64 *)&v39, (__int64 *)&v40, (__int64 *)&v41, (__int64 *)&v42);
    goto LABEL_9;
  }
  if ( *(_BYTE *)v15 != 85 )
  {
    v21 = (char *)sub_22B0AE0(v45, (__int64)&v45[(unsigned int)v46]);
    v22 = *(unsigned __int8 **)(a1 + 16);
    v42 = v21;
    v41 = sub_22AE640(*((_QWORD *)v22 + 1));
    v40 = sub_22AE640((unsigned int)*v22 - 29);
    v19 = sub_22B2950((__int64 *)&v40, (__int64 *)&v41, (__int64 *)&v42);
    goto LABEL_9;
  }
  v23 = *(_QWORD *)(v15 - 32);
  if ( !v23 || *(_BYTE *)v23 || *(_QWORD *)(v23 + 24) != *(_QWORD *)(v15 + 80) || (*(_BYTE *)(v23 + 33) & 0x20) == 0 )
  {
    v24 = *(_BYTE **)(a1 + 88);
    v25 = *(_QWORD *)(a1 + 96);
    v26 = (char *)v44;
    v42 = (char *)v44;
    if ( &v24[v25] && !v24 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v41 = v25;
    if ( v25 > 0xF )
    {
      v42 = (char *)sub_22409D0((__int64)&v42, &v41, 0);
      v36 = v42;
      v44[0] = v41;
    }
    else
    {
      if ( v25 == 1 )
      {
        LOBYTE(v44[0]) = *v24;
LABEL_22:
        v43 = v25;
        v26[v25] = 0;
        v41 = sub_22B0AE0(v45, (__int64)&v45[(unsigned int)v46]);
        v27 = sub_B3B940(v42, &v42[v43]);
        v28 = *(unsigned __int8 **)(a1 + 16);
        v40 = v27;
        v29 = *((_QWORD *)v28 + 1);
        v39 = sub_22AE640(v29);
        v38 = sub_22AE640(v29);
        v37 = sub_22AE640((unsigned int)*v28 - 29);
        v19 = sub_22B24A0((__int64 *)&v37, (__int64 *)&v38, (__int64 *)&v39, (__int64 *)&v40, (__int64 *)&v41);
        sub_2240A30((unsigned __int64 *)&v42);
        goto LABEL_9;
      }
      if ( !v25 )
        goto LABEL_22;
      v36 = (char *)v44;
    }
    memcpy(v36, v24, v25);
    v25 = v41;
    v26 = v42;
    goto LABEL_22;
  }
  v30 = *(_DWORD *)(v23 + 36);
  v31 = (char *)sub_22B0AE0(v45, (__int64)&v45[(unsigned int)v46]);
  v32 = *(char **)(a1 + 88);
  v33 = *(_QWORD *)(a1 + 96);
  v42 = v31;
  v41 = sub_B3B940(v32, &v32[v33]);
  v34 = sub_22AE640(v30);
  v35 = *(unsigned __int8 **)(a1 + 16);
  v40 = v34;
  v39 = sub_22AE640(*((_QWORD *)v35 + 1));
  v38 = sub_22AE640((unsigned int)*v35 - 29);
  v19 = sub_22B24A0((__int64 *)&v38, (__int64 *)&v39, (__int64 *)&v40, (__int64 *)&v41, (__int64 *)&v42);
LABEL_9:
  if ( v45 != (_QWORD *)v47 )
    _libc_free((unsigned __int64)v45);
  return v19;
}
