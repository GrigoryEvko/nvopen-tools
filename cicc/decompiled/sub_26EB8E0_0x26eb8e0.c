// Function: sub_26EB8E0
// Address: 0x26eb8e0
//
bool __fastcall sub_26EB8E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  bool result; // al
  char *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r15
  __int64 *v7; // rdi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rbx
  void *v12; // rax
  __int64 *v13; // rcx
  _BYTE **v14; // r12
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // r15
  unsigned __int64 v17; // r14
  unsigned __int64 v18; // rdx
  _QWORD *v19; // r9
  __int64 v20; // r13
  _QWORD *v21; // rbx
  _BYTE **v22; // r8
  _QWORD *v23; // r12
  __int64 v24; // r9
  unsigned __int64 v25; // r13
  unsigned __int64 v26; // rcx
  size_t v27; // rdx
  int v28; // eax
  _QWORD *v29; // r9
  __int64 *v30; // rax
  __int64 *v31; // rbx
  char v32; // al
  unsigned __int64 v33; // r15
  __int64 *v34; // r9
  __int64 ***v35; // rax
  __int64 *v36; // rdx
  size_t v37; // r13
  void *v38; // rax
  _QWORD *v39; // rsi
  unsigned __int64 v40; // rdi
  _QWORD *v41; // rcx
  unsigned __int64 v42; // rdx
  __int64 *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // [rsp+0h] [rbp-70h]
  __int64 v46; // [rsp+8h] [rbp-68h]
  __int64 v47; // [rsp+10h] [rbp-60h]
  bool v48; // [rsp+18h] [rbp-58h]
  _BYTE **v49; // [rsp+18h] [rbp-58h]
  __int64 *v50; // [rsp+18h] [rbp-58h]
  const void *v51[2]; // [rsp+20h] [rbp-50h] BYREF
  _BYTE v52[64]; // [rsp+30h] [rbp-40h] BYREF

  v2 = a2;
  result = sub_B2FC80(a2);
  if ( result )
    return 0;
  if ( (*(_BYTE *)(a2 + 32) & 0xF) != 1 )
  {
    if ( !byte_4FF8928 && (unsigned int)sub_2207590((__int64)&byte_4FF8928) )
    {
      v6 = qword_4FF8AE8;
      qword_4FF8948 = 1;
      qword_4FF8940 = (__int64)&qword_4FF8970;
      v7 = &qword_4FF8970 - 2;
      v46 = qword_4FF8AF0;
      qword_4FF8950 = 0;
      qword_4FF8958 = 0;
      v8 = (qword_4FF8AF0 - qword_4FF8AE8) >> 5;
      dword_4FF8960 = 1065353216;
      qword_4FF8968 = 0;
      qword_4FF8970 = 0;
      v9 = sub_222D860((__int64)(&qword_4FF8970 - 2), v8);
      v11 = v9;
      if ( v9 > qword_4FF8948 )
      {
        if ( v9 == 1 )
        {
          qword_4FF8970 = 0;
          v13 = &qword_4FF8970;
        }
        else
        {
          if ( v9 > 0xFFFFFFFFFFFFFFFLL )
LABEL_61:
            sub_4261EA(v7, v8, v10);
          v12 = (void *)sub_22077B0(8 * v9);
          v13 = (__int64 *)memset(v12, 0, 8 * v11);
        }
        qword_4FF8940 = (__int64)v13;
        qword_4FF8948 = v11;
      }
      if ( v46 != v6 )
      {
        v45 = v2;
        v14 = (_BYTE **)v6;
        do
        {
          v15 = sub_22076E0((__int64 *)*v14, (__int64)v14[1], 3339675911LL);
          v16 = qword_4FF8948;
          v17 = v15;
          v18 = v15 % qword_4FF8948;
          v19 = *(_QWORD **)(qword_4FF8940 + 8 * (v15 % qword_4FF8948));
          v20 = 8 * (v15 % qword_4FF8948);
          if ( !v19 )
            goto LABEL_34;
          v21 = (_QWORD *)*v19;
          v22 = v14;
          v23 = *(_QWORD **)(qword_4FF8940 + 8 * v18);
          v24 = 8 * v18;
          v25 = v15 % qword_4FF8948;
          v26 = v21[5];
          while ( 1 )
          {
            if ( v26 == v17 )
            {
              v27 = (size_t)v22[1];
              if ( v27 == v21[2] )
              {
                if ( !v27
                  || (v47 = v24, v49 = v22, v28 = memcmp(*v22, (const void *)v21[1], v27), v22 = v49, v24 = v47, !v28) )
                {
                  v20 = v24;
                  v29 = v23;
                  v14 = v22;
                  if ( !*v29 )
                    goto LABEL_34;
                  goto LABEL_29;
                }
              }
            }
            if ( !*v21 )
              break;
            v26 = *(_QWORD *)(*v21 + 40LL);
            v23 = v21;
            if ( v25 != v26 % v16 )
              break;
            v21 = (_QWORD *)*v21;
          }
          v20 = v24;
          v14 = v22;
LABEL_34:
          v30 = (__int64 *)sub_22077B0(0x30u);
          v31 = v30;
          if ( v30 )
            *v30 = 0;
          v30[1] = (__int64)(v30 + 3);
          sub_26E91F0(v30 + 1, *v14, (__int64)&v14[1][(_QWORD)*v14]);
          v8 = qword_4FF8948;
          v7 = (__int64 *)&dword_4FF8960;
          v32 = sub_222DA10((__int64)&dword_4FF8960, qword_4FF8948, qword_4FF8958, 1);
          v33 = v10;
          if ( !v32 )
          {
            v34 = (__int64 *)qword_4FF8940;
            goto LABEL_38;
          }
          if ( v10 == 1 )
          {
            qword_4FF8970 = 0;
            v34 = &qword_4FF8970;
            goto LABEL_44;
          }
          if ( v10 > 0xFFFFFFFFFFFFFFFLL )
            goto LABEL_61;
          v37 = 8 * v10;
          v38 = (void *)sub_22077B0(8 * v10);
          v34 = (__int64 *)memset(v38, 0, v37);
LABEL_44:
          v39 = (_QWORD *)qword_4FF8950;
          qword_4FF8950 = 0;
          if ( v39 )
          {
            v40 = 0;
            do
            {
              v41 = v39;
              v39 = (_QWORD *)*v39;
              v42 = v41[5] % v33;
              v43 = &v34[v42];
              if ( *v43 )
              {
                *v41 = *(_QWORD *)*v43;
                *(_QWORD *)*v43 = v41;
              }
              else
              {
                *v41 = qword_4FF8950;
                qword_4FF8950 = (__int64)v41;
                *v43 = (__int64)&qword_4FF8950;
                if ( *v41 )
                  v34[v40] = (__int64)v41;
                v40 = v42;
              }
            }
            while ( v39 );
          }
          if ( (__int64 *)qword_4FF8940 != &qword_4FF8970 )
          {
            v50 = v34;
            j_j___libc_free_0(qword_4FF8940);
            v34 = v50;
          }
          qword_4FF8948 = v33;
          qword_4FF8940 = (__int64)v34;
          v20 = 8 * (v17 % v33);
LABEL_38:
          v35 = (__int64 ***)((char *)v34 + v20);
          v31[5] = v17;
          v36 = *(__int64 **)((char *)v34 + v20);
          if ( v36 )
          {
            *v31 = *v36;
            **v35 = v31;
          }
          else
          {
            v44 = qword_4FF8950;
            qword_4FF8950 = (__int64)v31;
            *v31 = v44;
            if ( v44 )
            {
              v34[*(_QWORD *)(v44 + 40) % (unsigned __int64)qword_4FF8948] = (__int64)v31;
              v35 = (__int64 ***)(v20 + qword_4FF8940);
            }
            *v35 = (__int64 **)&qword_4FF8950;
          }
          ++qword_4FF8958;
LABEL_29:
          v14 += 4;
        }
        while ( (_BYTE **)v46 != v14 );
        v2 = v45;
      }
      __cxa_atexit((void (*)(void *))sub_8565C0, &qword_4FF8940, &qword_4A427C0);
      sub_2207640((__int64)&byte_4FF8928);
    }
    result = 1;
    if ( qword_4FF8958 )
    {
      v4 = (char *)sub_BD5D20(v2);
      if ( v4 )
      {
        v51[0] = v52;
        sub_26E9140((__int64 *)v51, v4, (__int64)&v4[v5]);
      }
      else
      {
        v51[1] = 0;
        v51[0] = v52;
        v52[0] = 0;
      }
      result = sub_BB97F0(&qword_4FF8940, v51) != 0;
      if ( v51[0] != v52 )
      {
        v48 = result;
        j_j___libc_free_0((unsigned __int64)v51[0]);
        return v48;
      }
    }
  }
  return result;
}
