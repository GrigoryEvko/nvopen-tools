// Function: sub_384F4A0
// Address: 0x384f4a0
//
unsigned __int64 __fastcall sub_384F4A0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r12
  char v5; // bl
  unsigned __int64 result; // rax
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // r14
  __int64 v10; // rdx
  int v11; // eax
  int v12; // eax
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  unsigned __int64 *v15; // r12
  int v16; // edx
  __int64 v17; // rdx
  int v18; // eax
  int v19; // edx
  __int64 v20; // rdx
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // rdi
  __int64 *v25; // r15
  _QWORD *v26; // rdi
  unsigned __int64 v27; // r14
  __int64 v28; // rsi
  char v29; // al
  __int64 v30; // rdx
  int v31; // eax
  int v32; // edx
  _QWORD *v33; // rdi
  __int64 v34; // rax
  _QWORD *v35; // rdi
  __int64 v36; // rdx
  int v37; // eax
  int v38; // edx
  int v39; // edx
  __int64 v40; // rax
  __int64 v41; // rdx
  int v42; // edx
  int v43; // [rsp+4h] [rbp-6Ch]
  unsigned int v44; // [rsp+8h] [rbp-68h]
  char v45; // [rsp+8h] [rbp-68h]
  unsigned __int64 v46; // [rsp+8h] [rbp-68h]
  int v47; // [rsp+10h] [rbp-60h]
  int v48; // [rsp+14h] [rbp-5Ch]
  int v50; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v51; // [rsp+30h] [rbp-40h] BYREF
  __int64 v52[7]; // [rsp+38h] [rbp-38h] BYREF

  v4 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = a2;
  if ( *(_BYTE *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 16) == 29 )
  {
    result = sub_157EBA0(*(_QWORD *)(v4 - 48));
    if ( *(_BYTE *)(result + 16) != 31 )
    {
      v7 = *(_QWORD *)(v4 + 40);
      goto LABEL_4;
    }
LABEL_17:
    *(_DWORD *)(a1 + 72) = 0;
    return result;
  }
  v7 = *(_QWORD *)(v4 + 40);
  result = sub_157EBA0(v7);
  if ( *(_BYTE *)(result + 16) == 31 )
    goto LABEL_17;
LABEL_4:
  v8 = *(_QWORD *)(v7 + 56);
  v9 = v8 + 112;
  if ( (unsigned __int8)sub_1560180(v8 + 112, 17) )
  {
    v10 = *(_QWORD *)(a1 + 64);
    v11 = *(_DWORD *)(a1 + 72);
    if ( *(_BYTE *)(v10 + 32) )
    {
      v16 = *(_DWORD *)(v10 + 28);
      if ( v11 > v16 )
        v11 = v16;
    }
    *(_DWORD *)(a1 + 72) = v11;
    v47 = 0;
    v48 = 0;
  }
  else
  {
    if ( (unsigned __int8)sub_1560180(v8 + 112, 34) || (unsigned __int8)sub_1560180(v8 + 112, 17) )
    {
      v17 = *(_QWORD *)(a1 + 64);
      v18 = *(_DWORD *)(a1 + 72);
      if ( *(_BYTE *)(v17 + 24) )
      {
        v19 = *(_DWORD *)(v17 + 20);
        if ( v18 > v19 )
          v18 = v19;
      }
      *(_DWORD *)(a1 + 72) = v18;
    }
    v47 = 150;
    v48 = 50;
  }
  v44 = 15000;
  if ( (unsigned __int8)sub_1560180(v8 + 112, 17) )
    goto LABEL_8;
  if ( (unsigned __int8)sub_1560180(a3 + 112, 15) )
  {
    v20 = *(_QWORD *)(a1 + 64);
    v21 = *(_DWORD *)(a1 + 72);
    if ( *(_BYTE *)(v20 + 8) )
    {
      v39 = *(_DWORD *)(v20 + 4);
      if ( v21 < v39 )
        v21 = v39;
    }
    *(_DWORD *)(a1 + 72) = v21;
  }
  v22 = *(_QWORD *)(a1 + 16);
  if ( *(_BYTE *)(v22 + 16) )
  {
    v23 = (*(__int64 (__fastcall **)(_QWORD, __int64))v22)(*(_QWORD *)(v22 + 8), v8);
    v24 = *(_QWORD **)(a1 + 24);
    v52[0] = a2;
    v25 = (__int64 *)v23;
    if ( !v24 || !(unsigned __int8)sub_1441AE0(v24) || !sub_1442240(*(_QWORD *)(a1 + 24), v52, v25) )
    {
      if ( v25 )
      {
        if ( *(_BYTE *)(*(_QWORD *)(a1 + 64) + 48LL) )
        {
          v46 = sub_1368AA0(v25, *(_QWORD *)((v52[0] & 0xFFFFFFFFFFFFFFF8LL) + 40));
          if ( v46 >= sub_1368DC0((__int64)v25) * dword_5051940 )
          {
            v40 = *(_QWORD *)(a1 + 64);
            v45 = *(_BYTE *)(v40 + 48);
            if ( v45 )
            {
              v45 = 1;
              v43 = *(_DWORD *)(v40 + 44);
            }
            goto LABEL_38;
          }
        }
LABEL_37:
        v45 = 0;
        goto LABEL_38;
      }
LABEL_55:
      v25 = 0;
      goto LABEL_37;
    }
  }
  else
  {
    v33 = *(_QWORD **)(a1 + 24);
    v52[0] = a2;
    if ( !v33 )
      goto LABEL_55;
    if ( !(unsigned __int8)sub_1441AE0(v33) )
      goto LABEL_55;
    v25 = 0;
    if ( !sub_1442240(*(_QWORD *)(a1 + 24), v52, 0) )
      goto LABEL_55;
  }
  v34 = *(_QWORD *)(a1 + 64);
  v45 = *(_BYTE *)(v34 + 40);
  if ( v45 )
  {
    v45 = 1;
    v43 = *(_DWORD *)(v34 + 36);
  }
LABEL_38:
  if ( !(unsigned __int8)sub_1560180(v9, 34) && !(unsigned __int8)sub_1560180(v9, 17) && v45 )
  {
    v44 = 15000;
    *(_DWORD *)(a1 + 72) = v43;
    goto LABEL_8;
  }
  v26 = *(_QWORD **)(a1 + 24);
  v51 = a2;
  if ( v26 && (unsigned __int8)sub_1441AE0(v26) )
  {
    v29 = sub_1442290(*(_QWORD *)(a1 + 24), &v51, v25);
  }
  else
  {
    if ( !v25 )
      goto LABEL_56;
    sub_16AF710(&v50, dword_5051A20, 0x64u);
    v27 = sub_1368AA0(v25, *(_QWORD *)((v51 & 0xFFFFFFFFFFFFFFF8LL) + 40));
    v28 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)((v51 & 0xFFFFFFFFFFFFFFF8LL) + 40) + 56LL) + 80LL);
    if ( v28 )
      v28 -= 24;
    v52[0] = sub_1368AA0(v25, v28);
    v29 = sub_16AF500(v52, v50) > v27;
  }
  if ( v29 )
  {
    v30 = *(_QWORD *)(a1 + 64);
    v31 = *(_DWORD *)(a1 + 72);
    if ( *(_BYTE *)(v30 + 56) )
    {
      v32 = *(_DWORD *)(v30 + 52);
      if ( v31 > v32 )
        v31 = v32;
    }
LABEL_49:
    *(_DWORD *)(a1 + 72) = v31;
    v44 = 0;
    v47 = 0;
    v48 = 0;
    goto LABEL_8;
  }
LABEL_56:
  v35 = *(_QWORD **)(a1 + 24);
  v44 = 15000;
  if ( v35 )
  {
    if ( sub_1441D10(v35, a3) )
    {
      v36 = *(_QWORD *)(a1 + 64);
      v37 = *(_DWORD *)(a1 + 72);
      if ( *(_BYTE *)(v36 + 8) )
      {
        v38 = *(_DWORD *)(v36 + 4);
        if ( v37 < v38 )
          v37 = v38;
      }
      *(_DWORD *)(a1 + 72) = v37;
      v44 = 15000;
      goto LABEL_8;
    }
    if ( sub_1441DA0(*(_QWORD **)(a1 + 24), a3) )
    {
      v41 = *(_QWORD *)(a1 + 64);
      v31 = *(_DWORD *)(a1 + 72);
      if ( *(_BYTE *)(v41 + 16) )
      {
        v42 = *(_DWORD *)(v41 + 12);
        if ( v31 > v42 )
          v31 = v42;
      }
      goto LABEL_49;
    }
  }
LABEL_8:
  v12 = *(_DWORD *)(a1 + 72) * sub_14A26B0(*(_QWORD *)a1);
  *(_DWORD *)(a1 + 72) = v12;
  *(_DWORD *)(a1 + 128) = v12 * v48 / 100;
  *(_DWORD *)(a1 + 120) = v47 * v12 / 100;
  v13 = *(_QWORD *)(a1 + 32);
  result = (*(_BYTE *)(v13 + 32) & 0xFu) - 7;
  if ( (unsigned int)result <= 1 )
  {
    result = *(_QWORD *)(v13 + 8);
    if ( result )
    {
      if ( !*(_QWORD *)(result + 8) )
      {
        v14 = v4 - 24;
        v15 = (unsigned __int64 *)(v4 - 72);
        if ( (v5 & 4) != 0 )
          v15 = (unsigned __int64 *)v14;
        result = *v15;
        if ( !*(_BYTE *)(*v15 + 16) && v13 == result )
        {
          *(_DWORD *)(a1 + 76) -= v44;
          return v44;
        }
      }
    }
  }
  return result;
}
