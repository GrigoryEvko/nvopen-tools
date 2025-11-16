// Function: sub_68FEF0
// Address: 0x68fef0
//
__int64 __fastcall sub_68FEF0(_QWORD *a1, _QWORD *a2, _DWORD *a3, int a4, __int64 a5, __int64 a6)
{
  _QWORD *v11; // rsi
  __int64 v12; // rdi
  __int64 i; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // rax
  int v19; // r14d
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  int v23; // edx
  __int64 v24; // rax
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r8
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v38; // [rsp+8h] [rbp-48h]
  int v39; // [rsp+14h] [rbp-3Ch] BYREF
  __int64 v40[7]; // [rsp+18h] [rbp-38h] BYREF

  v39 = 0;
  if ( (unsigned int)sub_68FE10(a1, 0, 1) || (v11 = 0, v12 = (__int64)a2, (unsigned int)sub_68FE10(a2, 0, 1)) )
  {
    v11 = 0;
    v12 = 34;
    sub_84EC30(34, 0, 0, 1, 0, (_DWORD)a1, (__int64)a2, (__int64)a3, a4, 0, 0, a6, a5, 0, (__int64)&v39);
  }
  if ( !v39 )
  {
    if ( !*((_BYTE *)a1 + 16) )
      goto LABEL_16;
    v17 = *a1;
    for ( i = *(unsigned __int8 *)(*a1 + 140LL); (_BYTE)i == 12; i = *(unsigned __int8 *)(v17 + 140) )
      v17 = *(_QWORD *)(v17 + 160);
    if ( !(_BYTE)i || !*((_BYTE *)a2 + 16) )
      goto LABEL_16;
    v18 = *a2;
    for ( i = *(unsigned __int8 *)(*a2 + 140LL); (_BYTE)i == 12; i = *(unsigned __int8 *)(v18 + 140) )
      v18 = *(_QWORD *)(v18 + 160);
    if ( !(_BYTE)i )
      goto LABEL_16;
    sub_6F69D0(a1, 0);
    sub_6F69D0(a2, 0);
    if ( (unsigned int)sub_8D2D80(*a1) && (unsigned int)sub_8D2D80(*a2)
      || (unsigned int)sub_8D28B0(*a1) && (unsigned int)sub_8D2780(*a2)
      || (unsigned int)sub_8D28B0(*a2) && (unsigned int)sub_8D2780(*a1) )
    {
      v19 = sub_8D29A0(*a1);
      if ( v19 != (unsigned int)sub_8D29A0(*a2) )
      {
        v11 = a3;
        v12 = 2917;
        sub_6861A0(0xB65u, a3, *a1, *a2);
        goto LABEL_16;
      }
      if ( *(_BYTE *)(*a1 + 140LL) == 2 && *(_BYTE *)(*a2 + 140LL) == 2 )
        sub_68B0C0((_DWORD *)*a1, *a2, (__int64)a3, 8u);
      v40[0] = sub_6E8B10(a1, a2, v20, v21, v22);
      sub_68B270(a1, v40[0], v31, v32, v33);
      sub_68B270(a2, v40[0], v34, v35, v36);
      sub_6FC7D0(v40[0], a1, a2, 64);
      if ( (unsigned int)sub_8D2930(v40[0]) )
        goto LABEL_29;
      if ( (unsigned int)sub_8D2AC0(v40[0]) )
      {
        v28 = sub_72CCD0();
        goto LABEL_30;
      }
    }
    else
    {
      if ( (unsigned int)sub_8D28B0(*a1) && (*a1 == *a2 || (unsigned int)sub_8D97D0(*a1, *a2, 0, v26, v27)) )
        goto LABEL_29;
      if ( (unsigned int)sub_8D2E30(*a1) || (unsigned int)sub_8D2E30(*a2) )
      {
        v11 = a2;
        v12 = (__int64)a1;
        if ( (unsigned int)sub_6EB6C0((_DWORD)a1, (_DWORD)a2, (_DWORD)a3, 34, 0, 0, 0, 0, (__int64)v40) )
        {
          sub_6FC7D0(v40[0], a1, a2, 64);
          if ( !(unsigned int)sub_8D2340(v40[0]) )
          {
LABEL_29:
            v28 = sub_72CC70();
LABEL_30:
            v38 = v28;
            v29 = sub_6F6F40(a1, 0);
            *(_QWORD *)(v29 + 16) = sub_6F6F40(a2, 0);
            v30 = sub_73DBF0(64, v38, v29);
            sub_6E70E0(v30, a6);
            goto LABEL_17;
          }
          goto LABEL_41;
        }
LABEL_16:
        sub_6E6000(v12, v11, i, v14, v15, v16);
        sub_6E6450(a1);
        sub_6E6450(a2);
        sub_6E6260(a6);
        goto LABEL_17;
      }
      if ( !(unsigned int)sub_8D3D10(*a1) && !(unsigned int)sub_8D3D10(*a2) && !(unsigned int)sub_8D2660(*a1) )
        sub_8D2660(*a2);
    }
LABEL_41:
    v11 = a3;
    v12 = 2917;
    sub_6E5ED0(2917, a3, *a1, *a2);
    goto LABEL_16;
  }
LABEL_17:
  v23 = *((_DWORD *)a1 + 17);
  *(_WORD *)(a6 + 72) = *((_WORD *)a1 + 36);
  *(_DWORD *)(a6 + 68) = v23;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a6 + 68);
  v24 = *(_QWORD *)((char *)a2 + 76);
  *(_QWORD *)(a6 + 76) = v24;
  unk_4F061D8 = v24;
  return sub_6E3280(a6, a3);
}
