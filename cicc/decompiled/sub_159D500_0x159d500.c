// Function: sub_159D500
// Address: 0x159d500
//
__int64 __fastcall sub_159D500(__int64 a1)
{
  __int16 v1; // r14
  __int64 *v2; // r12
  unsigned __int8 v3; // al
  __int16 v4; // r15
  __int64 v5; // r11
  __int64 v6; // r10
  __int64 *v7; // r9
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // r12
  __int64 v13; // rax
  unsigned int v14; // ebx
  __int64 v16; // rdx
  __int64 v17; // [rsp+8h] [rbp-1E8h]
  __int64 *v18; // [rsp+10h] [rbp-1E0h]
  __int64 v19; // [rsp+18h] [rbp-1D8h]
  __int16 v20; // [rsp+2Ch] [rbp-1C4h]
  char v21; // [rsp+2Fh] [rbp-1C1h]
  unsigned __int64 v22; // [rsp+30h] [rbp-1C0h] BYREF
  __int64 v23[4]; // [rsp+38h] [rbp-1B8h] BYREF
  __int64 v24; // [rsp+58h] [rbp-198h]
  __int64 v25; // [rsp+60h] [rbp-190h]
  __int64 v26; // [rsp+70h] [rbp-180h] BYREF
  char v27; // [rsp+78h] [rbp-178h] BYREF
  char v28; // [rsp+79h] [rbp-177h] BYREF
  __int16 v29; // [rsp+7Ah] [rbp-176h] BYREF
  __int64 *v30; // [rsp+80h] [rbp-170h]
  __int64 v31; // [rsp+88h] [rbp-168h]
  __int64 *v32; // [rsp+90h] [rbp-160h]
  __int64 v33; // [rsp+98h] [rbp-158h]
  __int64 *v34; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v35; // [rsp+B8h] [rbp-138h]
  _BYTE v36[304]; // [rsp+C0h] [rbp-130h] BYREF

  v1 = 0;
  v2 = (__int64 *)a1;
  v35 = 0x2000000000LL;
  v3 = *(_BYTE *)(a1 + 17);
  v34 = (__int64 *)v36;
  v4 = *(_WORD *)(a1 + 18);
  v21 = v3 >> 1;
  if ( sub_1594520(a1) )
  {
    v1 = sub_1594720(a1);
    if ( !sub_1594700(a1) )
      goto LABEL_3;
  }
  else if ( !sub_1594700(a1) )
  {
LABEL_3:
    v5 = 0;
    v6 = 0;
    v7 = 0;
    goto LABEL_4;
  }
  v24 = sub_1594710(a1);
  v7 = (__int64 *)v24;
  v25 = v16;
  v6 = v16;
  v5 = v24 + 4 * v16;
LABEL_4:
  v8 = (unsigned int)v35;
  v9 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( v9 )
  {
    v20 = v1;
    v10 = 0;
    v11 = v9 - 1;
    while ( 1 )
    {
      v12 = *(_QWORD *)(a1 + 24 * (v10 - v9));
      if ( HIDWORD(v35) <= (unsigned int)v8 )
      {
        v17 = v5;
        v18 = v7;
        v19 = v6;
        sub_16CD150(&v34, v36, 0, 8);
        v8 = (unsigned int)v35;
        v5 = v17;
        v7 = v18;
        v6 = v19;
      }
      v34[v8] = v12;
      v8 = (unsigned int)(v35 + 1);
      LODWORD(v35) = v35 + 1;
      if ( v11 == v10 )
        break;
      ++v10;
      v9 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    }
    v2 = (__int64 *)a1;
    v1 = v20;
  }
  v13 = *v2;
  v32 = v7;
  v33 = v6;
  v26 = v13;
  v29 = v1;
  v28 = v21;
  v27 = v4;
  v30 = v34;
  v31 = (unsigned int)v8;
  v23[0] = sub_1597510(v7, v5);
  v22 = sub_1597240(v30, (__int64)&v30[v31]);
  LODWORD(v23[0]) = sub_1597150(&v27, &v28, &v29, (__int64 *)&v22, v23);
  v14 = sub_15981B0(&v26, (int *)v23);
  if ( v34 != (__int64 *)v36 )
    _libc_free((unsigned __int64)v34);
  return v14;
}
