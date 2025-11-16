// Function: sub_26CC080
// Address: 0x26cc080
//
__int64 __fastcall sub_26CC080(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, unsigned __int64 *a5, __int64 a6)
{
  __int64 v6; // rax
  int *v7; // r15
  size_t v8; // r14
  _QWORD *v9; // rax
  __int64 v10; // rdi
  unsigned int v11; // r15d
  __int64 v12; // rbx
  const char *v13; // r15
  __int64 v14; // rdx
  __int64 *v15; // rbx
  __int64 *v16; // r13
  int v17; // r14d
  __int64 v18; // r12
  unsigned int v20; // eax
  __int64 v21; // rsi
  const char *v22; // rax
  __int64 v23; // rdx
  _BYTE *v24; // rax
  _BYTE *v25; // r13
  char v26; // al
  unsigned int v27; // eax
  __int64 v28; // rax
  float v29; // xmm0_4
  float v30; // xmm1_4
  __int64 v31; // rdx
  __int64 v32; // [rsp+18h] [rbp-128h]
  _QWORD *v37; // [rsp+40h] [rbp-100h]
  __int64 v40; // [rsp+58h] [rbp-E8h]
  _QWORD v41[2]; // [rsp+60h] [rbp-E0h] BYREF
  __int64 *v42; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v43; // [rsp+78h] [rbp-C8h]
  _QWORD v44[24]; // [rsp+80h] [rbp-C0h] BYREF

  v6 = a3[1];
  v7 = *(int **)(v6 + 16);
  v8 = *(_QWORD *)(v6 + 24);
  if ( v7 )
  {
    sub_C7D030(&v42);
    sub_C7D280((int *)&v42, v7, v8);
    sub_C7D290(&v42, v41);
    v8 = v41[0];
  }
  v42 = (__int64 *)v8;
  v9 = sub_26C56D0((_QWORD *)(a1 + 1296), (__int64 *)&v42);
  v37 = v9;
  if ( !v9 )
    return 0;
  v10 = v9[2];
  v11 = 0;
  if ( v10 )
  {
    v12 = *a3;
    v32 = *a3;
    v13 = sub_BD5D20(v10);
    v40 = v14;
    v41[0] = 0;
    sub_ED2710((__int64)&v42, v12, 0, qword_4FF62C8, v41, 1u);
    if ( (_DWORD)v43 )
    {
      v15 = v42;
      v16 = &v42[2 * (unsigned int)v43];
      v17 = 0;
      while ( 1 )
      {
        if ( v15[1] == -1 )
        {
          v18 = *v15;
          if ( v18 == sub_B2F650((__int64)v13, v40) )
            break;
          if ( ++v17 == (_DWORD)qword_4FF62C8 )
            break;
        }
        v15 += 2;
        if ( v16 == v15 )
          goto LABEL_14;
      }
      if ( v42 != v44 )
      {
        _libc_free((unsigned __int64)v42);
        return 0;
      }
      return 0;
    }
LABEL_14:
    if ( v42 != v44 )
      _libc_free((unsigned __int64)v42);
    v41[0] = "Callee function not available";
    LOBYTE(v20) = sub_B2FC80(v37[2]);
    v11 = v20;
    if ( (_BYTE)v20 )
      return 0;
    if ( !sub_B92180(v37[2]) )
      return 0;
    if ( !(unsigned __int8)sub_B2D620(v37[2], "use-sample-profile", 0x12u) )
      return 0;
    v21 = v37[2];
    if ( v21 == a2 || !(unsigned __int8)sub_29A3A40(v32, v21, v41) )
      return 0;
    v22 = sub_BD5D20(v37[2]);
    v44[0] = sub_B2F650((__int64)v22, v23);
    v43 = 0x100000001LL;
    v42 = v44;
    v44[1] = -1;
    sub_26CB7A0(v32, (__int64)&v42, 0);
    v24 = (_BYTE *)sub_2445EC0(v32, (unsigned __int8 *)v37[2], a3[2], *a5, 0, *(__int64 **)(a1 + 1288));
    v25 = v24;
    if ( v24 )
    {
      *a5 -= a3[2];
      *a3 = (__int64)v24;
      v26 = *v24;
      if ( v26 == 85 || v26 == 34 )
      {
        if ( *(_BYTE *)(a1 + 1705) || (v27 = sub_26C3F00(a1, (__int64)a3, a6), !(_BYTE)v27) )
        {
          v28 = a3[2];
          if ( v28 < 0 )
          {
            v31 = a3[2] & 1 | ((unsigned __int64)a3[2] >> 1);
            v29 = (float)(int)v31 + (float)(int)v31;
          }
          else
          {
            v29 = (float)(int)v28;
          }
          if ( a4 < 0 )
            v30 = (float)(a4 & 1 | (unsigned int)((unsigned __int64)a4 >> 1))
                + (float)(a4 & 1 | (unsigned int)((unsigned __int64)a4 >> 1));
          else
            v30 = (float)(int)a4;
          sub_3144140(v25, v29 / v30);
        }
        else
        {
          v11 = v27;
        }
      }
    }
    if ( v42 != v44 )
      _libc_free((unsigned __int64)v42);
  }
  return v11;
}
