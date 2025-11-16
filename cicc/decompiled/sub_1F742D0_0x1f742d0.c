// Function: sub_1F742D0
// Address: 0x1f742d0
//
__int64 __fastcall sub_1F742D0(__int64 a1, __int64 a2, unsigned int a3, unsigned int *a4, unsigned int a5)
{
  char v6; // r15
  unsigned int v9; // eax
  char v10; // di
  __int64 v11; // rax
  unsigned int v12; // eax
  __int16 v13; // dx
  __int64 v14; // r15
  char v15; // al
  unsigned int v16; // r8d
  __int64 v18; // r15
  __int64 v19; // rax
  char v20; // di
  __int64 v21; // rax
  unsigned __int8 v22; // r12
  int v23; // eax
  __int64 v24; // rax
  __int64 v25; // rdx
  char v26; // al
  __int64 v27; // rdx
  __int64 v28; // rax
  char v29; // di
  __int64 v30; // rax
  unsigned __int8 v31; // r8
  unsigned int v32; // r15d
  int v33; // eax
  __int64 v34; // rdi
  __int64 (*v35)(); // rax
  unsigned int v36; // eax
  unsigned int v37; // [rsp+8h] [rbp-48h]
  unsigned int v38; // [rsp+8h] [rbp-48h]
  unsigned __int8 v39; // [rsp+8h] [rbp-48h]
  unsigned int v41; // [rsp+Ch] [rbp-44h]
  char v42[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v43; // [rsp+18h] [rbp-38h]

  if ( !a2 || (a5 & 7) != 0 )
    return 0;
  v6 = *(_BYTE *)a4;
  v9 = *(_BYTE *)a4 ? sub_1F6C8D0(*(_BYTE *)a4) : sub_1F58D40((__int64)a4);
  if ( v9 <= 7 || (v9 & (v9 - 1)) != 0 || (*(_BYTE *)(a2 + 26) & 8) != 0 )
    return 0;
  v10 = *(_BYTE *)(a2 + 88);
  v11 = *(_QWORD *)(a2 + 96);
  v42[0] = v10;
  v43 = v11;
  if ( v10 )
  {
    v37 = sub_1F6C8D0(v10);
    if ( v6 )
      goto LABEL_10;
LABEL_19:
    v12 = sub_1F58D40((__int64)a4);
    goto LABEL_11;
  }
  v37 = sub_1F58D40((__int64)v42);
  if ( !v6 )
    goto LABEL_19;
LABEL_10:
  v12 = sub_1F6C8D0(v6);
LABEL_11:
  if ( v12 > v37 )
    return 0;
  if ( a5 )
  {
    v18 = *(_QWORD *)(a1 + 8);
    v38 = sub_1E340A0(*(_QWORD *)(a2 + 104));
    v19 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)a1 + 32LL));
    if ( !(unsigned __int8)sub_1F43CC0(
                             v18,
                             *(_QWORD *)(*(_QWORD *)a1 + 48LL),
                             v19,
                             *a4,
                             *((_QWORD *)a4 + 1),
                             v38,
                             a5 >> 3,
                             0) )
      return 0;
  }
  v13 = *(_WORD *)(a2 + 24);
  v14 = *(_QWORD *)(a2 + 32);
  if ( v13 == 186 )
  {
    v26 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v14 + 80) + 40LL) + 16LL * *(unsigned int *)(v14 + 88));
    if ( v26 == 113 || !v26 )
      return 0;
LABEL_23:
    v20 = *(_BYTE *)(a2 + 88);
    v21 = *(_QWORD *)(a2 + 96);
    v42[0] = v20;
    v43 = v21;
    if ( v20 )
      v41 = sub_1F6C8D0(v20);
    else
      v41 = sub_1F58D40((__int64)v42);
    v22 = *(_BYTE *)a4;
    if ( *(_BYTE *)a4 )
      v23 = sub_1F6C8D0(*(_BYTE *)a4);
    else
      v23 = sub_1F58D40((__int64)a4);
    if ( v23 + a5 > v41 )
      return 0;
    if ( *(_BYTE *)(a1 + 24) )
    {
      v24 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v14 + 40) + 40LL) + 16LL * *(unsigned int *)(v14 + 48));
      if ( (_BYTE)v24 )
      {
        v25 = *(_QWORD *)(a1 + 8);
        if ( *(_QWORD *)(v25 + 8 * v24 + 120) )
        {
          if ( v22 )
          {
            LOBYTE(v16) = *(_BYTE *)(v22 + 115LL * (unsigned __int8)v24 + v25 + 58658) == 0;
            return v16;
          }
        }
      }
      return 0;
    }
    return 1;
  }
  v15 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v14 + 40) + 40LL) + 16LL * *(unsigned int *)(v14 + 48));
  if ( v15 == 0 || v15 == 113 )
    return 0;
  if ( v13 != 185 )
    goto LABEL_23;
  if ( !sub_1D18C00(a2, 1, 0) )
    return 0;
  if ( *(_BYTE *)(a1 + 24) )
  {
    v27 = *(unsigned __int8 *)a4;
    v28 = **(unsigned __int8 **)(a2 + 40);
    if ( !(_BYTE)v28
      || !(_BYTE)v27
      || (((int)*(unsigned __int16 *)(*(_QWORD *)(a1 + 8) + 2 * (v27 + 115 * v28 + 16104)) >> (4 * a3)) & 0xF) != 0 )
    {
      return 0;
    }
  }
  if ( *(_DWORD *)(a2 + 60) > 2u )
    return 0;
  if ( (*(_BYTE *)(a2 + 27) & 0xC) != 0 )
  {
    v29 = *(_BYTE *)(a2 + 88);
    v30 = *(_QWORD *)(a2 + 96);
    v42[0] = v29;
    v43 = v30;
    if ( v29 )
    {
      v32 = sub_1F6C8D0(v29);
    }
    else
    {
      v36 = sub_1F58D40((__int64)v42);
      v31 = 0;
      v32 = v36;
    }
    if ( *(_BYTE *)a4 )
    {
      v33 = sub_1F6C8D0(*(_BYTE *)a4);
    }
    else
    {
      v39 = v31;
      v33 = sub_1F58D40((__int64)a4);
      v16 = v39;
    }
    if ( a5 + v33 > v32 )
      return v16;
  }
  v34 = *(_QWORD *)(a1 + 8);
  v35 = *(__int64 (**)())(*(_QWORD *)v34 + 416LL);
  if ( v35 == sub_1F3CAB0 )
    return 1;
  return ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD, _QWORD))v35)(v34, a2, a3, *a4, *((_QWORD *)a4 + 1));
}
