// Function: sub_32067D0
// Address: 0x32067d0
//
__int64 __fastcall sub_32067D0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rdx
  unsigned __int8 *v4; // r13
  unsigned int v5; // r12d
  unsigned __int64 v6; // r14
  unsigned __int8 v7; // al
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int8 v10; // al
  bool v11; // dl
  int v12; // ebx
  int v13; // ebx
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 v16; // r15
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 *v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // rax
  bool v22; // al
  unsigned __int64 v23; // rcx
  unsigned __int8 v24; // al
  const char *v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  signed __int64 v29; // rax
  unsigned __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // rsi
  unsigned __int64 v33; // rax
  __int64 *v34; // rdi
  unsigned int v35; // eax
  __int64 *v36; // rax
  unsigned int v37; // edx
  __int64 v39; // [rsp+8h] [rbp-C8h]
  _BOOL8 v40; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v41; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v44; // [rsp+38h] [rbp-98h]
  __int64 v45; // [rsp+38h] [rbp-98h]
  __int64 v46; // [rsp+58h] [rbp-78h]
  int v47; // [rsp+6Ch] [rbp-64h]
  __int16 v48; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v49; // [rsp+72h] [rbp-5Eh]
  int v50; // [rsp+76h] [rbp-5Ah]
  unsigned __int64 v51; // [rsp+80h] [rbp-50h]
  const char *v52; // [rsp+88h] [rbp-48h]
  __int64 v53; // [rsp+90h] [rbp-40h]

  v2 = *(_BYTE *)(a2 - 16);
  v39 = a2 - 16;
  if ( (v2 & 2) != 0 )
    v3 = *(_QWORD *)(a2 - 32);
  else
    v3 = v39 - 8LL * ((v2 >> 2) & 0xF);
  v4 = *(unsigned __int8 **)(v3 + 24);
  v5 = sub_3206530(a1, v4, 0);
  if ( sub_AE2980(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL) + 312LL, 0)[1] >> 3 == 8 )
    v47 = 35;
  else
    v47 = 34;
  v6 = (unsigned __int64)sub_3212020(v4) >> 3;
  v7 = *(_BYTE *)(a2 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *(_QWORD *)(a2 - 32);
  else
    v8 = v39 - 8LL * ((v7 >> 2) & 0xF);
  v9 = *(_QWORD *)(v8 + 32);
  v46 = v9;
  if ( v9 )
  {
    v10 = *(_BYTE *)(v9 - 16);
    v11 = (v10 & 2) != 0;
    v12 = (v10 & 2) != 0 ? *(_DWORD *)(v46 - 24) : (*(_WORD *)(v46 - 16) >> 6) & 0xF;
    v13 = v12 - 1;
    if ( v13 >= 0 )
    {
      v14 = 8LL * v13;
      while ( 1 )
      {
        if ( v11 )
          v15 = *(_QWORD *)(v46 - 32);
        else
          v15 = v46 - 16 - 8LL * ((v10 >> 2) & 0xF);
        v16 = *(_QWORD *)(v15 + v14);
        v17 = sub_AF2780(v16);
        if ( v17 && (v17 & 6) == 0 && (v18 = v17 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v19 = *(__int64 **)(v18 + 24);
          v20 = *(_DWORD *)(v18 + 32);
          if ( v20 > 0x40 )
          {
            v21 = *v19;
          }
          else
          {
            if ( !v20 )
              goto LABEL_33;
            v21 = (__int64)((_QWORD)v19 << (64 - (unsigned __int8)v20)) >> (64 - (unsigned __int8)v20);
          }
        }
        else
        {
          v29 = sub_AF2880(v16);
          if ( !v29 )
            goto LABEL_33;
          v45 = (v29 >> 1) & 3;
          if ( ((v29 >> 1) & 3) != 0 )
            goto LABEL_33;
          v41 = v29 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v29 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            goto LABEL_33;
          v40 = *(_BYTE *)(a1 + 800) == 2;
          v30 = sub_AF2800(v16);
          v31 = v40;
          v32 = v45;
          if ( v30 )
          {
            if ( (v30 & 6) == 0 )
            {
              v33 = v30 & 0xFFFFFFFFFFFFFFF8LL;
              if ( v33 )
              {
                v34 = *(__int64 **)(v33 + 24);
                v35 = *(_DWORD *)(v33 + 32);
                if ( v35 > 0x40 )
                {
                  v31 = *v34;
                }
                else
                {
                  v31 = 0;
                  if ( v35 )
                    v31 = (__int64)((_QWORD)v34 << (64 - (unsigned __int8)v35)) >> (64 - (unsigned __int8)v35);
                }
              }
            }
          }
          v36 = *(__int64 **)(v41 + 24);
          v37 = *(_DWORD *)(v41 + 32);
          if ( v37 > 0x40 )
          {
            v32 = *v36;
          }
          else if ( v37 )
          {
            v32 = (__int64)((_QWORD)v36 << (64 - (unsigned __int8)v37)) >> (64 - (unsigned __int8)v37);
          }
          v21 = v32 - v31 + 1;
        }
        if ( v21 != -1 )
        {
          v6 *= v21;
          v22 = v6 == 0;
          goto LABEL_21;
        }
LABEL_33:
        v6 = 0;
        v22 = 1;
LABEL_21:
        if ( v13 )
        {
          v23 = v6;
          v27 = 0;
          v25 = byte_3F871B3;
        }
        else
        {
          if ( v22 )
            v23 = *(_QWORD *)(a2 + 24) >> 3;
          else
            v23 = v6;
          v24 = *(_BYTE *)(a2 - 16);
          if ( (v24 & 2) != 0 )
          {
            v25 = *(const char **)(*(_QWORD *)(a2 - 32) + 16LL);
            if ( v25 )
              goto LABEL_26;
          }
          else
          {
            v25 = *(const char **)(v39 - 8LL * ((v24 >> 2) & 0xF) + 16);
            if ( v25 )
            {
LABEL_26:
              v44 = v23;
              v26 = sub_B91420((__int64)v25);
              v23 = v44;
              v25 = (const char *)v26;
              goto LABEL_27;
            }
          }
          v27 = 0;
        }
LABEL_27:
        v52 = v25;
        --v13;
        v14 -= 8;
        v48 = 5379;
        v49 = v5;
        v51 = v23;
        v50 = v47;
        v53 = v27;
        v28 = sub_3709C80(a1 + 648, &v48);
        v5 = sub_3707F80(a1 + 632, v28);
        if ( v13 == -1 )
          return v5;
        v10 = *(_BYTE *)(v46 - 16);
        v11 = (v10 & 2) != 0;
      }
    }
  }
  return v5;
}
