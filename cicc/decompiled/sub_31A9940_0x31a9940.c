// Function: sub_31A9940
// Address: 0x31a9940
//
__int64 __fastcall sub_31A9940(__int64 a1)
{
  __int64 v1; // rax
  __int64 *v2; // rbx
  __int64 *v3; // rax
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rsi
  __int64 *v16; // rax
  __int64 *v17; // r14
  __int64 v18; // rbx
  unsigned __int64 v19; // rax
  __int64 v20; // r13
  int v21; // eax
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned __int64 v24; // rax
  int v25; // edx
  __int64 v26; // rax
  char v27; // al
  __int64 v29; // r12
  int v30; // edi
  unsigned int i; // r15d
  __int64 v32; // rsi
  _QWORD *v33; // rax
  _QWORD *v34; // rdx
  unsigned __int64 v35; // rax
  int v36; // edx
  __int64 v37; // rax
  __int64 v38; // r13
  unsigned __int8 v39; // al
  __int64 v40; // rsi
  __int64 *v41; // rax
  unsigned __int8 v42; // [rsp+Fh] [rbp-F1h]
  __int64 v43; // [rsp+18h] [rbp-E8h]
  __int64 v44; // [rsp+28h] [rbp-D8h]
  __int64 *v45; // [rsp+28h] [rbp-D8h]
  __int64 *v46; // [rsp+30h] [rbp-D0h]
  _QWORD *v47; // [rsp+30h] [rbp-D0h]
  _BYTE *v49; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+48h] [rbp-B8h]
  _BYTE v51[32]; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+70h] [rbp-90h] BYREF
  __int64 *v53; // [rsp+78h] [rbp-88h]
  __int64 v54; // [rsp+80h] [rbp-80h]
  int v55; // [rsp+88h] [rbp-78h]
  char v56; // [rsp+8Ch] [rbp-74h]
  char v57; // [rsp+90h] [rbp-70h] BYREF

  v42 = qword_50355E8;
  v1 = *(_QWORD *)a1;
  if ( !(_BYTE)qword_50355E8 )
  {
    sub_2AB8760(
      (__int64)"If-conversion is disabled",
      25,
      "If-conversion is disabled",
      0x19u,
      (__int64)"IfConversionDisabled",
      20,
      *(__int64 **)(a1 + 64),
      v1,
      0);
    return v42;
  }
  v2 = *(__int64 **)(v1 + 32);
  v3 = *(__int64 **)(v1 + 40);
  v52 = 0;
  v53 = (__int64 *)&v57;
  v54 = 8;
  v55 = 0;
  v56 = 1;
  v46 = v3;
  if ( v3 != v2 )
  {
    while ( 1 )
    {
      v4 = *v2;
      v5 = *v2 + 48;
      if ( !sub_31A6C30(a1, *v2) )
        break;
      v10 = *(_QWORD *)(v4 + 56);
      v44 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 112LL);
      v49 = v51;
      v50 = 0x400000000LL;
      if ( v10 != v5 )
      {
        while ( 1 )
        {
          if ( !v10 )
            goto LABEL_79;
          if ( *(_BYTE *)(v10 - 24) == 61
            && (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 - 16) + 8LL) - 17 > 1
            && !(unsigned __int8)sub_D30ED0(v10 - 24)
            && (unsigned __int8)sub_D30800(
                                  v10 - 24,
                                  *(_QWORD *)a1,
                                  v44,
                                  *(__int64 **)(a1 + 40),
                                  *(_QWORD *)(a1 + 432),
                                  (__int64)&v49) )
          {
            v15 = *(_QWORD *)(v10 - 56);
            if ( !v56 )
              goto LABEL_69;
            v16 = v53;
            v12 = HIDWORD(v54);
            v11 = &v53[HIDWORD(v54)];
            if ( v53 != v11 )
            {
              while ( v15 != *v16 )
              {
                if ( v11 == ++v16 )
                  goto LABEL_16;
              }
              goto LABEL_6;
            }
LABEL_16:
            if ( HIDWORD(v54) < (unsigned int)v54 )
            {
              ++HIDWORD(v54);
              *v11 = v15;
              ++v52;
            }
            else
            {
LABEL_69:
              sub_C8CC70((__int64)&v52, v15, (__int64)v11, v12, v13, v14);
            }
          }
LABEL_6:
          LODWORD(v50) = 0;
          v10 = *(_QWORD *)(v10 + 8);
          if ( v10 == v5 )
          {
            if ( v49 != v51 )
              _libc_free((unsigned __int64)v49);
            break;
          }
        }
      }
LABEL_20:
      if ( v46 == ++v2 )
      {
        v45 = *(__int64 **)(*(_QWORD *)a1 + 40LL);
        if ( *(__int64 **)(*(_QWORD *)a1 + 32LL) != v45 )
        {
          v17 = *(__int64 **)(*(_QWORD *)a1 + 32LL);
          v43 = a1 + 440;
          do
          {
            v18 = *v17;
            v47 = (_QWORD *)(*v17 + 48);
            v19 = *v47 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v47 == (_QWORD *)v19 )
              goto LABEL_38;
            if ( !v19 )
              goto LABEL_79;
            v20 = v19 - 24;
            v21 = *(unsigned __int8 *)(v19 - 24);
            if ( (unsigned int)(v21 - 30) > 0xA )
LABEL_38:
              BUG();
            if ( (_BYTE)v21 == 32 )
            {
              v29 = *(_QWORD *)a1;
              v30 = sub_B46E30(v20);
              if ( v30 )
              {
                for ( i = 0; v30 != i; ++i )
                {
                  v32 = sub_B46EC0(v20, i);
                  if ( *(_BYTE *)(v29 + 84) )
                  {
                    v33 = *(_QWORD **)(v29 + 64);
                    v34 = &v33[*(unsigned int *)(v29 + 76)];
                    if ( v33 == v34 )
                      goto LABEL_49;
                    while ( v32 != *v33 )
                    {
                      if ( v34 == ++v33 )
                        goto LABEL_49;
                    }
                  }
                  else if ( !sub_C8CA60(v29 + 56, v32) )
                  {
LABEL_49:
                    v35 = *(_QWORD *)(v18 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                    if ( v47 == (_QWORD *)v35 )
                    {
                      v37 = 0;
                      goto LABEL_71;
                    }
                    if ( v35 )
                    {
                      v36 = *(unsigned __int8 *)(v35 - 24);
                      v37 = v35 - 24;
                      if ( (unsigned int)(v36 - 30) >= 0xB )
                        v37 = 0;
LABEL_71:
                      sub_2AB8760(
                        (__int64)"Loop contains an unsupported switch",
                        35,
                        "Loop contains an unsupported switch",
                        0x23u,
                        (__int64)"LoopContainsUnsupportedSwitch",
                        29,
                        *(__int64 **)(a1 + 64),
                        *(_QWORD *)a1,
                        v37);
                      v42 = 0;
                      v27 = v56;
LABEL_36:
                      if ( v27 )
                        return v42;
LABEL_72:
                      _libc_free((unsigned __int64)v53);
                      return v42;
                    }
LABEL_79:
                    BUG();
                  }
                }
              }
            }
            else if ( (_BYTE)v21 != 31 )
            {
              sub_2AB8760(
                (__int64)"Loop contains an unsupported terminator",
                39,
                "Loop contains an unsupported terminator",
                0x27u,
                (__int64)"LoopContainsUnsupportedTerminator",
                33,
                *(__int64 **)(a1 + 64),
                *(_QWORD *)a1,
                v20);
              v42 = 0;
              if ( v56 )
                return v42;
              goto LABEL_72;
            }
            if ( sub_31A6C30(a1, v18) && !(unsigned __int8)sub_31A9620(a1, v18, &v52, v43, v22, v23) )
            {
              v24 = *(_QWORD *)(v18 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v47 == (_QWORD *)v24 )
              {
                v26 = 0;
              }
              else
              {
                if ( !v24 )
                  goto LABEL_79;
                v25 = *(unsigned __int8 *)(v24 - 24);
                v26 = v24 - 24;
                if ( (unsigned int)(v25 - 30) >= 0xB )
                  v26 = 0;
              }
              sub_2AB8760(
                (__int64)"Control flow cannot be substituted for a select",
                47,
                "Control flow cannot be substituted for a select",
                0x2Fu,
                (__int64)"NoCFGForSelect",
                14,
                *(__int64 **)(a1 + 64),
                *(_QWORD *)a1,
                v26);
              v42 = 0;
              v27 = v56;
              goto LABEL_36;
            }
            ++v17;
          }
          while ( v45 != v17 );
        }
        if ( v56 )
          return v42;
        goto LABEL_72;
      }
    }
    v38 = *(_QWORD *)(v4 + 56);
    if ( v38 == v5 )
      goto LABEL_20;
    while ( 1 )
    {
      if ( !v38 )
        goto LABEL_79;
      v39 = *(_BYTE *)(v38 - 24);
      if ( v39 > 0x1Cu && (v39 == 61 || v39 == 62) )
      {
        v40 = *(_QWORD *)(v38 - 56);
        if ( v40 )
        {
          if ( !v56 )
            goto LABEL_68;
          v41 = v53;
          v7 = HIDWORD(v54);
          v6 = &v53[HIDWORD(v54)];
          if ( v53 != v6 )
          {
            while ( *v41 != v40 )
            {
              if ( v6 == ++v41 )
                goto LABEL_66;
            }
            goto LABEL_57;
          }
LABEL_66:
          if ( HIDWORD(v54) < (unsigned int)v54 )
          {
            v7 = (unsigned int)++HIDWORD(v54);
            *v6 = v40;
            ++v52;
          }
          else
          {
LABEL_68:
            sub_C8CC70((__int64)&v52, v40, (__int64)v6, v7, v8, v9);
          }
        }
      }
LABEL_57:
      v38 = *(_QWORD *)(v38 + 8);
      if ( v38 == v5 )
        goto LABEL_20;
    }
  }
  return v42;
}
