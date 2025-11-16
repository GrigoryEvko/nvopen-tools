// Function: sub_801880
// Address: 0x801880
//
__int64 __fastcall sub_801880(int a1)
{
  int v1; // r10d
  _QWORD *v2; // rbx
  _QWORD *v3; // rcx
  _QWORD *v4; // rsi
  _QWORD *v5; // r14
  _BYTE *v6; // rdx
  __int64 v7; // rax
  _QWORD *v8; // r13
  _BYTE *v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdi
  _QWORD *v17; // r15
  __int64 v18; // r12
  char v19; // al
  char v20; // al
  char *v21; // r13
  int v22; // eax
  __int64 result; // rax
  __int64 v24; // r11
  __int64 v25; // r11
  __int64 v26; // rcx
  __int64 v27; // r8
  _QWORD *v28; // r12
  _BYTE *v29; // [rsp+0h] [rbp-1E0h]
  __int64 *v30; // [rsp+8h] [rbp-1D8h]
  unsigned int v31; // [rsp+14h] [rbp-1CCh]
  _QWORD *v32; // [rsp+18h] [rbp-1C8h]
  __int64 v33; // [rsp+20h] [rbp-1C0h]
  __int64 v34; // [rsp+28h] [rbp-1B8h]
  _QWORD *v35; // [rsp+28h] [rbp-1B8h]
  __int64 v36; // [rsp+30h] [rbp-1B0h]
  __int64 v37; // [rsp+38h] [rbp-1A8h]
  unsigned int v38; // [rsp+44h] [rbp-19Ch] BYREF
  __int64 v39; // [rsp+48h] [rbp-198h] BYREF
  __m128i v40[2]; // [rsp+50h] [rbp-190h] BYREF
  __m128i v41[2]; // [rsp+70h] [rbp-170h] BYREF
  __m128i v42[5]; // [rsp+90h] [rbp-150h] BYREF
  _BYTE v43[32]; // [rsp+E0h] [rbp-100h] BYREF
  void (__fastcall *v44)(__int64); // [rsp+100h] [rbp-E0h]

  v1 = a1;
  v37 = qword_4F07288;
  v2 = *(_QWORD **)(qword_4F07288 + 192);
  if ( a1 )
  {
    v32 = 0;
    v3 = 0;
    if ( v2 )
    {
      v4 = 0;
      v5 = 0;
      v6 = v43;
      while ( 1 )
      {
        v7 = v2[1];
        v8 = v2;
        v2 = (_QWORD *)*v2;
        *v8 = 0;
        if ( (*(_BYTE *)(v7 + 170) & 0x60) != 0 || *(_BYTE *)(v7 + 177) == 5 )
          goto LABEL_5;
        if ( *(char *)(v7 + 172) >= 0 && a1 == *(unsigned __int16 *)(v7 + 158) )
        {
          if ( v4 )
            *v4 = v8;
          else
            v3 = v8;
          v35 = v3;
          v36 = (__int64)v6;
          sub_76C7C0((__int64)v6);
          v44 = sub_7F5240;
          sub_76D400((__int64)v8, v36, v36, v26, v27);
          v6 = (_BYTE *)v36;
          v3 = v35;
          v4 = v8;
          goto LABEL_5;
        }
        if ( v5 )
        {
          *v5 = v8;
          v5 = v8;
LABEL_5:
          if ( !v2 )
            goto LABEL_12;
        }
        else
        {
          v32 = v8;
          v5 = v8;
          if ( !v2 )
          {
LABEL_12:
            v1 = a1;
            v9 = v6;
            goto LABEL_13;
          }
        }
      }
    }
    v9 = v43;
LABEL_13:
    v10 = *(_QWORD *)(v37 + 88);
    v33 = v10;
    if ( v10 )
    {
      v11 = *(_QWORD *)(v10 + 24);
      if ( v11 )
      {
        v33 = 0;
        v12 = 0;
        v13 = 0;
        v14 = 0;
        while ( 1 )
        {
          v15 = v11;
          v11 = *(_QWORD *)(v11 + 32);
          *(_QWORD *)(v15 + 32) = 0;
          if ( (*(_BYTE *)(v15 + 50) & 8) != 0 )
            break;
          if ( v13 )
          {
            *(_QWORD *)(v13 + 32) = v15;
            v13 = v15;
            if ( !v11 )
            {
LABEL_22:
              v10 = *(_QWORD *)(v37 + 88);
              *(_QWORD *)(v37 + 192) = v3;
              if ( v10 )
                goto LABEL_23;
              goto LABEL_24;
            }
          }
          else
          {
            v33 = v15;
            v13 = v15;
LABEL_18:
            if ( !v11 )
              goto LABEL_22;
          }
        }
        if ( v14 )
        {
          *(_QWORD *)(v14 + 32) = v15;
          v14 = v15;
        }
        else
        {
          v14 = v15;
          v12 = v15;
        }
        goto LABEL_18;
      }
      v12 = 0;
      *(_QWORD *)(v37 + 192) = v3;
      v33 = 0;
LABEL_23:
      *(_QWORD *)(v10 + 24) = v12;
LABEL_24:
      v31 = 1;
      v2 = v3;
    }
    else
    {
      v2 = v3;
      v31 = 1;
      *(_QWORD *)(v37 + 192) = v3;
    }
  }
  else
  {
    v31 = 0;
    v9 = v43;
    v33 = 0;
    v32 = 0;
  }
  v30 = sub_7F7930(0, v1, "__sti__", (__int64)v40, (int *)&v38, (__int64)v9, 0);
  dword_4D03EB8[0] = 1;
  v16 = *(_QWORD *)(v37 + 88);
  if ( v16 )
    sub_7E9190(v16, (__int64)v40);
  if ( v2 )
  {
    v29 = v9;
    while ( 1 )
    {
      v17 = v2;
      v2 = (_QWORD *)*v2;
      v18 = v17[1];
      *v17 = 0;
      v19 = *(_BYTE *)(v18 + 170);
      if ( (v19 & 0x60) == 0 && *(_BYTE *)(v18 + 177) != 5 && *(char *)(v18 + 172) >= 0 )
      {
        if ( (*(_QWORD *)(v18 + 168) & 0x2000100000LL) == 0 || (*(_BYTE *)(v18 + 88) & 0x70) == 0x10 || v19 < 0 )
        {
          v20 = *(_BYTE *)(v18 + 156);
          if ( (v20 & 1) != 0 )
            goto LABEL_35;
          sub_7F9080(v18, (__int64)v42);
          sub_7E2BA0((__int64)v40);
          sub_7FEC50((__int64)v17, v42, 0, 0, 1, 0, v40, 0, 0);
          sub_7FAFA0((__int64)v40);
        }
        else
        {
          sub_7F8CF0(v18, v40[0].m128i_i32, v41, 0, &v39);
          v24 = v39;
          *(_BYTE *)(v39 + 168) = *(_BYTE *)(v18 + 168) & 7 | *(_BYTE *)(v39 + 168) & 0xF8;
          *(_BYTE *)(v24 + 176) = *(_BYTE *)(v18 + 176) & 8 | *(_BYTE *)(v24 + 176) & 0xF7;
          v20 = *(_BYTE *)(v17[1] + 156LL);
          if ( (v20 & 1) != 0 )
          {
            v18 = v17[1];
LABEL_35:
            v21 = "__constant__";
            if ( (v20 & 4) == 0 )
            {
              v21 = "__managed__";
              if ( (*(_BYTE *)(v18 + 157) & 1) == 0 )
              {
                v21 = "__device__";
                if ( (v20 & 2) != 0 )
                  v21 = "__shared__";
              }
            }
            v22 = sub_7FA8C0((__int64)v17);
            sub_6849F0(v22 == 0 ? 8 : 4, 0xDBEu, (_DWORD *)(v18 + 64), (__int64)v21);
            goto LABEL_40;
          }
          v34 = v24;
          sub_7F9080(v18, (__int64)v42);
          sub_7E2BA0((__int64)v41);
          sub_7FEC50((__int64)v17, v42, 0, 0, 1, 0, v41, 0, 0);
          sub_7FAFA0((__int64)v41);
          v25 = v34;
          if ( v41[0].m128i_i32[0] == 1 )
          {
            v28 = sub_73E830(v18);
            sub_7304E0((__int64)v28);
            sub_7E69E0(v28, v41[0].m128i_i32);
            v25 = v34;
          }
          sub_7F6E70(v25, v41[0].m128i_i32);
        }
      }
LABEL_40:
      if ( !v2 )
      {
        v9 = v29;
        break;
      }
    }
  }
  sub_7FB010((__int64)v30, v38, (__int64)v9);
  dword_4D03EB8[0] = 0;
  result = v31;
  if ( v31 )
  {
    *(_QWORD *)(v37 + 192) = v32;
    result = *(_QWORD *)(v37 + 88);
    if ( result )
      *(_QWORD *)(result + 24) = v33;
  }
  return result;
}
