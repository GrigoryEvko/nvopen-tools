// Function: sub_33AD070
// Address: 0x33ad070
//
void __fastcall sub_33AD070(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 *v4; // r14
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // edx
  int v13; // eax
  int v14; // r15d
  _BYTE *v15; // rdx
  __int64 v16; // rax
  _BYTE *v17; // rdx
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r15
  int v22; // edx
  _QWORD *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r14
  int v27; // edx
  int v28; // r15d
  _QWORD *v29; // rax
  __int64 v30; // [rsp+0h] [rbp-100h]
  __int64 v31; // [rsp+8h] [rbp-F8h]
  int v32; // [rsp+10h] [rbp-F0h]
  int v33; // [rsp+18h] [rbp-E8h]
  __int64 v34; // [rsp+30h] [rbp-D0h]
  int v35; // [rsp+30h] [rbp-D0h]
  __int64 v36; // [rsp+38h] [rbp-C8h]
  __int64 v37; // [rsp+70h] [rbp-90h] BYREF
  __int64 v38; // [rsp+78h] [rbp-88h]
  __int64 v39; // [rsp+80h] [rbp-80h] BYREF
  int v40; // [rsp+88h] [rbp-78h]
  __int64 v41; // [rsp+90h] [rbp-70h] BYREF
  int v42; // [rsp+98h] [rbp-68h]
  _BYTE *v43; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v44; // [rsp+A8h] [rbp-58h]
  _BYTE v45[80]; // [rsp+B0h] [rbp-50h] BYREF

  v3 = *(_QWORD *)(a1 + 864);
  v4 = *(__int64 **)(a2 + 8);
  v5 = *(_QWORD *)(v3 + 16);
  v6 = sub_2E79000(*(__int64 **)(v3 + 40));
  v39 = 0;
  LODWORD(v37) = sub_2D5BAE0(v5, v6, v4, 0);
  v7 = *(_QWORD *)a1;
  v38 = v8;
  v40 = *(_DWORD *)(a1 + 848);
  if ( v7 )
  {
    if ( &v39 != (__int64 *)(v7 + 48) )
    {
      v9 = *(_QWORD *)(v7 + 48);
      v39 = v9;
      if ( v9 )
        sub_B96E90((__int64)&v39, v9, 1);
    }
  }
  v32 = sub_338B750(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v33 = v12;
  if ( (_WORD)v37 )
  {
    if ( (unsigned __int16)(v37 - 176) > 0x34u )
    {
      v43 = v45;
      v13 = word_4456340[(unsigned __int16)v37 - 1];
      v44 = 0x800000000LL;
      goto LABEL_10;
    }
  }
  else if ( !sub_3007100((__int64)&v37) )
  {
    v43 = v45;
    v44 = 0x800000000LL;
    v13 = sub_3007240((__int64)&v37);
LABEL_10:
    if ( v13 )
    {
      v14 = v13 - 1;
      v15 = v45;
      v16 = 0;
      while ( 1 )
      {
        *(_DWORD *)&v15[4 * v16] = v14;
        v16 = (unsigned int)(v44 + 1);
        LODWORD(v44) = v44 + 1;
        if ( !v14 )
          break;
        if ( v16 + 1 > (unsigned __int64)HIDWORD(v44) )
        {
          sub_C8D5F0((__int64)&v43, v45, v16 + 1, 4u, v10, v11);
          v16 = (unsigned int)v44;
        }
        v15 = v43;
        --v14;
      }
      v17 = v43;
    }
    else
    {
      v17 = v45;
      v16 = 0;
    }
    v18 = *(_QWORD *)(a1 + 864);
    v34 = (__int64)v17;
    v36 = v16;
    v41 = 0;
    v42 = 0;
    v19 = sub_33F17F0(v18, 51, &v41, v37, v38);
    if ( v41 )
    {
      v30 = v19;
      v31 = v20;
      sub_B91220((__int64)&v41, v41);
      v19 = v30;
      v20 = v31;
    }
    v21 = sub_33FCE10(v18, v37, v38, (unsigned int)&v39, v32, v33, v19, v20, v34, v36);
    v35 = v22;
    v41 = a2;
    v23 = sub_337DC20(a1 + 8, &v41);
    *v23 = v21;
    *((_DWORD *)v23 + 2) = v35;
    if ( v43 != v45 )
      _libc_free((unsigned __int64)v43);
    v24 = v39;
    if ( v39 )
      goto LABEL_22;
    return;
  }
  v25 = sub_33FAF80(*(_QWORD *)(a1 + 864), 164, (unsigned int)&v39, v37, v38, v11);
  v43 = (_BYTE *)a2;
  v26 = v25;
  v28 = v27;
  v29 = sub_337DC20(a1 + 8, (__int64 *)&v43);
  *v29 = v26;
  v24 = v39;
  *((_DWORD *)v29 + 2) = v28;
  if ( v24 )
LABEL_22:
    sub_B91220((__int64)&v39, v24);
}
