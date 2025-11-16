// Function: sub_17921E0
// Address: 0x17921e0
//
__int64 __fastcall sub_17921E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v8; // r8
  __int64 v9; // rax
  __int64 v10; // r14
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rcx
  unsigned __int8 v16; // al
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rsi
  _QWORD *v22; // r14
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // rcx
  __int64 v29; // r14
  unsigned __int8 v30; // al
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rsi
  unsigned __int8 v35; // al
  __int64 v36; // rax
  __int64 v37; // rdi
  _QWORD *v38; // r14
  __int64 v39; // rsi
  __int64 v40; // [rsp+0h] [rbp-70h]
  __int64 *v41; // [rsp+0h] [rbp-70h]
  __int64 v42; // [rsp+8h] [rbp-68h]
  __int64 *v43; // [rsp+8h] [rbp-68h]
  __int64 v44; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v45; // [rsp+18h] [rbp-58h]
  __int64 v46[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v47; // [rsp+30h] [rbp-40h]

  v8 = *(_BYTE *)(a4 + 16);
  if ( (unsigned __int8)(*(_BYTE *)(a3 + 16) - 35) <= 0x11u )
  {
    v9 = *(_QWORD *)(a3 + 8);
    if ( v9 )
    {
      if ( !*(_QWORD *)(v9 + 8) && v8 > 0x10u )
      {
        v24 = sub_1790570(a3);
        if ( v24 )
        {
          if ( (v24 & 1) != 0 )
          {
            v25 = *(_QWORD *)(a3 - 48);
            if ( v25 == a4 )
            {
              if ( v25 )
              {
                v27 = -24;
                goto LABEL_30;
              }
            }
          }
          if ( (v24 & 2) != 0 )
          {
            v26 = *(_QWORD *)(a3 - 24);
            if ( v26 == a4 )
            {
              v27 = -48;
              if ( v26 )
              {
LABEL_30:
                sub_17909C0((__int64)&v44, a3);
                v29 = *(_QWORD *)(a3 + v27);
                v30 = *(_BYTE *)(v29 + 16);
                v31 = v29 + 24;
                if ( v30 == 13 )
                {
LABEL_31:
                  if ( !(unsigned __int8)sub_1790AD0((__int64)&v44, v31) )
                    goto LABEL_32;
LABEL_51:
                  v36 = sub_15A1070(*(_QWORD *)v29, (__int64)&v44);
                  v37 = *(_QWORD *)(a1 + 8);
                  v47 = 257;
                  v38 = sub_1707C10(v37, *(_QWORD *)(a2 - 72), v29, v36, v46, 0);
                  sub_164B7C0((__int64)v38, a3);
                  v47 = 257;
                  v10 = sub_15FB440(
                          (unsigned int)*(unsigned __int8 *)(a3 + 16) - 24,
                          (__int64 *)a4,
                          (__int64)v38,
                          (__int64)v46,
                          0);
                  sub_15F2530((unsigned __int8 *)v10, a3, 1);
                  if ( v45 <= 0x40 )
                    return v10;
LABEL_19:
                  if ( v44 )
                    j_j___libc_free_0_0(v44);
                  return v10;
                }
                if ( *(_BYTE *)(*(_QWORD *)v29 + 8LL) == 16 )
                {
                  if ( v30 > 0x10u )
                    goto LABEL_51;
                  v39 = sub_15A1020((_BYTE *)v29, v31, *(_QWORD *)v29, v28);
                  if ( !v39 )
                  {
                    if ( *(_BYTE *)(v29 + 16) > 0x10u )
                      goto LABEL_51;
LABEL_32:
                    if ( v45 > 0x40 && v44 )
                      j_j___libc_free_0_0(v44);
                    v8 = *(_BYTE *)(a4 + 16);
                    goto LABEL_5;
                  }
                  v30 = *(_BYTE *)(v29 + 16);
                  if ( *(_BYTE *)(v39 + 16) == 13 )
                  {
                    v31 = v39 + 24;
                    if ( v30 > 0x10u )
                      goto LABEL_51;
                    goto LABEL_31;
                  }
                }
                if ( v30 > 0x10u )
                  goto LABEL_51;
                goto LABEL_32;
              }
            }
          }
        }
      }
    }
  }
LABEL_5:
  v10 = 0;
  if ( (unsigned __int8)(v8 - 35) <= 0x11u )
  {
    v10 = *(_QWORD *)(a4 + 8);
    if ( v10 )
    {
      v10 = *(_QWORD *)(v10 + 8);
      if ( v10 )
        return 0;
      if ( *(_BYTE *)(a3 + 16) > 0x10u )
      {
        v11 = sub_1790570(a4);
        if ( v11 )
        {
          if ( (v11 & 1) != 0 && (v32 = *(_QWORD *)(a4 - 48), v32 == a3) && v32 )
          {
            v13 = -24;
          }
          else
          {
            if ( (v11 & 2) == 0 )
              return v10;
            v12 = *(_QWORD *)(a4 - 24);
            if ( !v12 || v12 != a3 )
              return v10;
            v13 = -48;
          }
          v40 = v13;
          sub_17909C0((__int64)&v44, a4);
          v15 = *(__int64 **)(a4 + v40);
          v16 = *((_BYTE *)v15 + 16);
          v17 = (__int64)(v15 + 3);
          if ( v16 == 13 )
          {
LABEL_16:
            v41 = v15;
            if ( (unsigned __int8)sub_1790AD0((__int64)&v44, v17) )
            {
              v15 = v41;
              v18 = *v41;
              goto LABEL_18;
            }
            goto LABEL_41;
          }
          v18 = *v15;
          if ( *(_BYTE *)(*v15 + 8) != 16 )
          {
            if ( v16 <= 0x10u )
              goto LABEL_41;
LABEL_18:
            v42 = (__int64)v15;
            v19 = sub_15A1070(v18, (__int64)&v44);
            v20 = *(_QWORD *)(a1 + 8);
            v21 = *(_QWORD *)(a2 - 72);
            v47 = 257;
            v22 = sub_1707C10(v20, v21, v19, v42, v46, 0);
            sub_164B7C0((__int64)v22, a4);
            LODWORD(v20) = *(unsigned __int8 *)(a4 + 16) - 24;
            v47 = 257;
            v10 = sub_15FB440(v20, (__int64 *)a3, (__int64)v22, (__int64)v46, 0);
            sub_15F2530((unsigned __int8 *)v10, a4, 1);
            if ( v45 <= 0x40 )
              return v10;
            goto LABEL_19;
          }
          if ( v16 > 0x10u )
            goto LABEL_18;
          v43 = *(__int64 **)(a4 + v40);
          v33 = sub_15A1020(v43, v17, v14, (__int64)v15);
          v15 = v43;
          v34 = v33;
          if ( v33 )
          {
            v35 = *((_BYTE *)v43 + 16);
            if ( *(_BYTE *)(v34 + 16) == 13 )
            {
              v17 = v34 + 24;
              if ( v35 <= 0x10u )
                goto LABEL_16;
              goto LABEL_47;
            }
            if ( v35 > 0x10u )
            {
LABEL_47:
              v18 = *v43;
              goto LABEL_18;
            }
          }
          else if ( *((_BYTE *)v43 + 16) > 0x10u )
          {
            goto LABEL_47;
          }
LABEL_41:
          if ( v45 <= 0x40 )
            return v10;
          goto LABEL_19;
        }
      }
    }
  }
  return v10;
}
