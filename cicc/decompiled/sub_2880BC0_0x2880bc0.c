// Function: sub_2880BC0
// Address: 0x2880bc0
//
__int64 __fastcall sub_2880BC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 *v7; // rdi
  __int64 *v8; // rax
  __int64 *v9; // r15
  __int64 v12; // rsi
  int v13; // ecx
  bool v14; // zf
  char v15; // si
  int v16; // edx
  __int64 v17; // rax
  int v18; // ecx
  signed __int64 v19; // rax
  unsigned __int64 v20; // rdx
  signed __int64 v21; // rdx
  __int64 v23; // rax
  __int16 v25; // [rsp+20h] [rbp-C0h] BYREF
  char v26; // [rsp+22h] [rbp-BEh]
  int v27; // [rsp+24h] [rbp-BCh]
  char v28; // [rsp+28h] [rbp-B8h]
  __int64 v29; // [rsp+30h] [rbp-B0h]
  int v30; // [rsp+38h] [rbp-A8h]
  __int64 v31; // [rsp+40h] [rbp-A0h]
  int v32; // [rsp+48h] [rbp-98h]
  int v33; // [rsp+50h] [rbp-90h]
  __int64 v34; // [rsp+58h] [rbp-88h]
  __int64 v35; // [rsp+60h] [rbp-80h]
  __int64 v36; // [rsp+68h] [rbp-78h]
  unsigned int v37; // [rsp+70h] [rbp-70h]
  __int64 v38; // [rsp+78h] [rbp-68h]
  _QWORD v39[5]; // [rsp+80h] [rbp-60h] BYREF

  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  v7 = *(__int64 **)(a2 + 40);
  v25 = 0;
  v8 = *(__int64 **)(a2 + 32);
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  memset(v39, 0, sizeof(v39));
  if ( v7 == v8 )
  {
    v14 = (_BYTE)qword_5002A68 == 0;
    *(_DWORD *)(a1 + 20) = 0;
    if ( v14 )
    {
      *(_BYTE *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
LABEL_17:
      v23 = sub_D4A330(a2);
      v18 = v32;
      *(_BYTE *)(a1 + 28) = v23 == 0;
      if ( v18 != 1 )
        goto LABEL_7;
LABEL_18:
      *(_DWORD *)(a1 + 8) = 1;
      goto LABEL_8;
    }
    v16 = 0;
    v13 = 0;
    v15 = 0;
    goto LABEL_15;
  }
  v9 = v8;
  do
  {
    v12 = *v9++;
    sub_30ABD80(&v25, v12, a3, a4, 0, a2);
  }
  while ( v7 != v9 );
  v13 = HIDWORD(v39[3]);
  v14 = (_BYTE)qword_5002A68 == 0;
  v15 = v26;
  v16 = v27;
  *(_DWORD *)(a1 + 20) = HIDWORD(v39[3]);
  if ( !v14 )
LABEL_15:
    *(_DWORD *)(a1 + 20) = v13 + HIDWORD(v38) + LODWORD(v39[0]);
  v17 = v29;
  *(_BYTE *)(a1 + 16) = v15;
  *(_DWORD *)(a1 + 24) = v16;
  *(_QWORD *)a1 = v17;
  *(_DWORD *)(a1 + 8) = v30;
  if ( v16 != 3 )
    goto LABEL_17;
  v18 = v32;
  *(_BYTE *)(a1 + 28) = 0;
  if ( v18 == 1 )
    goto LABEL_18;
LABEL_7:
  v18 = *(_DWORD *)(a1 + 8);
LABEL_8:
  v19 = *(_QWORD *)a1 - v31;
  if ( __OFSUB__(*(_QWORD *)a1, v31) )
  {
    v19 = 0x8000000000000000LL;
    if ( v31 <= 0 )
      v19 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v20 = *(_QWORD *)((char *)v39 + 4);
  *(_QWORD *)a1 = v19;
  *(_OWORD *)(a1 + 32) = __PAIR128__(*(_QWORD *)((char *)&v39[1] + 4), v20);
  *(_QWORD *)(a1 + 48) = *(_QWORD *)((char *)&v39[2] + 4);
  if ( !v18 )
  {
    v21 = (unsigned int)(a5 + 1);
    if ( v21 > v19 )
    {
      *(_QWORD *)a1 = v21;
      *(_DWORD *)(a1 + 8) = 0;
    }
  }
  return sub_C7D6A0(v35, 24LL * v37, 8);
}
