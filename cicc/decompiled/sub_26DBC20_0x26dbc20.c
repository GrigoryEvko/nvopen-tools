// Function: sub_26DBC20
// Address: 0x26dbc20
//
__int64 __fastcall sub_26DBC20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // esi
  __int64 v13; // rdi
  int v14; // esi
  unsigned int v15; // ecx
  __int64 *v16; // rdx
  __int64 v17; // r8
  __int64 v18; // rax
  unsigned int v19; // ebx
  unsigned int v20; // eax
  __int64 result; // rax
  unsigned int v22; // edx
  int v23; // edx
  int v24; // r9d
  __int64 v25; // r13
  __int64 v26; // r14
  int v27; // esi
  __int64 v28; // r8
  __int64 *v29; // rdi
  unsigned int v30; // edx
  __int64 *v31; // rax
  __int64 *v32; // rax
  int v33; // edx
  __int64 v35; // [rsp+8h] [rbp-98h]
  unsigned __int64 v37; // [rsp+18h] [rbp-88h]
  __int64 v38; // [rsp+18h] [rbp-88h]
  __int64 v39; // [rsp+20h] [rbp-80h] BYREF
  __int64 *v40; // [rsp+28h] [rbp-78h] BYREF
  __int64 v41[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v42; // [rsp+40h] [rbp-60h]
  __int64 v43; // [rsp+50h] [rbp-50h] BYREF
  __int64 v44; // [rsp+58h] [rbp-48h]
  __int64 v45; // [rsp+60h] [rbp-40h]
  unsigned int v46; // [rsp+68h] [rbp-38h]

  v6 = a2 + 72;
  v7 = a2;
  v9 = *(_QWORD *)(a2 + 80);
  if ( LOBYTE(qword_500BA28[8]) )
  {
    v43 = 0;
    v44 = 0;
    v45 = 0;
    v46 = 0;
    if ( v9 == v6 )
    {
LABEL_39:
      v41[0] = v7;
      v41[1] = a1 + 1056;
      v42 = &v43;
      sub_26D93E0(v41, a1 + 40, a1 + 72, a4, (__int64)v41, a6);
      return sub_C7D6A0(v44, 16LL * v46, 8);
    }
    v25 = v9;
    while ( 1 )
    {
      while ( 1 )
      {
        v26 = v25 - 24;
        if ( !v25 )
          v26 = 0;
        sub_26C6CF0((__int64)v41, (void (__fastcall ***)(unsigned __int64 *, _QWORD, __int64))a1, v26);
        if ( ((unsigned __int8)v42 & 1) == 0 )
          break;
        v25 = *(_QWORD *)(v25 + 8);
        if ( v25 == v6 )
          goto LABEL_38;
      }
      v27 = v46;
      v28 = v41[0];
      v39 = v26;
      if ( !v46 )
        break;
      a4 = 1;
      v29 = 0;
      v30 = (v46 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v31 = (__int64 *)(v44 + 16LL * v30);
      a6 = *v31;
      if ( v26 != *v31 )
      {
        while ( a6 != -4096 )
        {
          if ( a6 == -8192 && !v29 )
            v29 = v31;
          v30 = (v46 - 1) & (a4 + v30);
          v31 = (__int64 *)(v44 + 16LL * v30);
          a6 = *v31;
          if ( v26 == *v31 )
            goto LABEL_36;
          a4 = (unsigned int)(a4 + 1);
        }
        if ( !v29 )
          v29 = v31;
        ++v43;
        v33 = v45 + 1;
        v40 = v29;
        if ( 4 * ((int)v45 + 1) < 3 * v46 )
        {
          a6 = v46 >> 3;
          if ( v46 - HIDWORD(v45) - v33 > (unsigned int)a6 )
          {
LABEL_50:
            LODWORD(v45) = v33;
            if ( *v29 != -4096 )
              --HIDWORD(v45);
            *v29 = v26;
            v32 = v29 + 1;
            v29[1] = 0;
            goto LABEL_37;
          }
          v35 = v41[0];
LABEL_55:
          sub_FE19E0((__int64)&v43, v27);
          sub_26C35D0((__int64)&v43, &v39, &v40);
          v26 = v39;
          v29 = v40;
          v28 = v35;
          v33 = v45 + 1;
          goto LABEL_50;
        }
LABEL_54:
        v35 = v41[0];
        v27 = 2 * v46;
        goto LABEL_55;
      }
LABEL_36:
      v32 = v31 + 1;
LABEL_37:
      *v32 = v28;
      v25 = *(_QWORD *)(v25 + 8);
      if ( v25 == v6 )
      {
LABEL_38:
        v7 = a2;
        goto LABEL_39;
      }
    }
    ++v43;
    v40 = 0;
    goto LABEL_54;
  }
  if ( v9 == v6 )
    goto LABEL_14;
  do
  {
    while ( 1 )
    {
      v10 = v9 - 24;
      v11 = *(_QWORD *)(a1 + 1016);
      if ( !v9 )
        v10 = 0;
      v41[0] = v10;
      v12 = *(_DWORD *)(v11 + 24);
      v13 = *(_QWORD *)(v11 + 8);
      if ( v12 )
      {
        v14 = v12 - 1;
        v15 = v14 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v16 = (__int64 *)(v13 + 16LL * v15);
        v17 = *v16;
        if ( v10 != *v16 )
        {
          v23 = 1;
          while ( v17 != -4096 )
          {
            v24 = v23 + 1;
            v15 = v14 & (v23 + v15);
            v16 = (__int64 *)(v13 + 16LL * v15);
            v17 = *v16;
            if ( v10 == *v16 )
              goto LABEL_9;
            v23 = v24;
          }
          goto LABEL_4;
        }
LABEL_9:
        v18 = v16[1];
        if ( v18 )
        {
          v43 = **(_QWORD **)(v18 + 32);
          if ( v43 )
          {
            v37 = *sub_26CC460(a1 + 40, v41);
            if ( v37 > *sub_26CC460(a1 + 40, &v43) )
              break;
          }
        }
      }
LABEL_4:
      v9 = *(_QWORD *)(v9 + 8);
      if ( v9 == v6 )
        goto LABEL_13;
    }
    v38 = *sub_26CC460(a1 + 40, v41);
    *sub_26CC460(a1 + 40, &v43) = v38;
    v9 = *(_QWORD *)(v9 + 8);
  }
  while ( v9 != v6 );
LABEL_13:
  v7 = a2;
LABEL_14:
  v19 = 0;
  do
    v20 = v19++;
  while ( LODWORD(qword_500BDA8[8]) > v20 && (unsigned __int8)sub_26DAB40(a1, v7, 0) );
  *(_DWORD *)(a1 + 400) = 0;
  sub_26BBBD0(*(_QWORD *)(a1 + 936));
  *(_QWORD *)(a1 + 936) = 0;
  *(_QWORD *)(a1 + 944) = a1 + 928;
  *(_QWORD *)(a1 + 952) = a1 + 928;
  *(_QWORD *)(a1 + 960) = 0;
  do
  {
    result = LODWORD(qword_500BDA8[8]);
    v22 = v19++;
    if ( LODWORD(qword_500BDA8[8]) <= v22 )
      goto LABEL_21;
  }
  while ( (unsigned __int8)sub_26DAB40(a1, v7, 0) );
  do
  {
    result = LODWORD(qword_500BDA8[8]);
LABEL_21:
    if ( v19 >= (unsigned int)result )
      break;
    ++v19;
    result = sub_26DAB40(a1, v7, 1);
  }
  while ( (_BYTE)result );
  return result;
}
