// Function: sub_14DB6D0
// Address: 0x14db6d0
//
__int64 __fastcall sub_14DB6D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // r8
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // r12
  _BYTE *v9; // r15
  int v10; // eax
  __int64 v11; // rax
  __int64 *v12; // r9
  __int64 *v13; // rbx
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 *v16; // r15
  __int64 v17; // r12
  char v18; // al
  __int64 v19; // rax
  _QWORD *v20; // r12
  __int64 v21; // rbx
  unsigned int v22; // eax
  __int64 v24; // rdx
  int v25; // esi
  unsigned int v26; // edi
  __int64 *v27; // rax
  __int64 v28; // r10
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  int v32; // eax
  int v33; // r9d
  _BYTE *v34; // [rsp+0h] [rbp-E0h]
  _BYTE *v35; // [rsp+8h] [rbp-D8h]
  _BYTE *v36; // [rsp+8h] [rbp-D8h]
  __int64 v37; // [rsp+8h] [rbp-D8h]
  __int64 v39; // [rsp+18h] [rbp-C8h]
  __int64 v40; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v41; // [rsp+28h] [rbp-B8h] BYREF
  _BYTE v42[48]; // [rsp+30h] [rbp-B0h] BYREF
  _BYTE *v43; // [rsp+60h] [rbp-80h] BYREF
  __int64 v44; // [rsp+68h] [rbp-78h]
  _BYTE v45[112]; // [rsp+70h] [rbp-70h] BYREF

  v4 = (_BYTE *)a4;
  v6 = a1;
  v7 = *(unsigned __int8 *)(a1 + 16);
  v39 = a2;
  if ( (_BYTE)v7 == 5 || (v8 = 0, (_BYTE)v7 == 8) )
  {
    v9 = v45;
    v44 = 0x800000000LL;
    v10 = *(_DWORD *)(a1 + 20);
    v43 = v45;
    v11 = 3LL * (v10 & 0xFFFFFFF);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    {
      v13 = *(__int64 **)(a1 - 8);
      v12 = &v13[v11];
    }
    else
    {
      v12 = (__int64 *)a1;
      v13 = (__int64 *)(a1 - v11 * 8);
    }
    if ( v12 != v13 )
    {
      v14 = a3;
      v15 = a4;
      v4 = v45;
      v16 = v12;
      while ( 1 )
      {
        v17 = *v13;
        v18 = *(_BYTE *)(*v13 + 16);
        if ( v18 != 5 && v18 != 8 )
          goto LABEL_9;
        a4 = *(_BYTE *)(v15 + 8) & 1;
        if ( (*(_BYTE *)(v15 + 8) & 1) != 0 )
        {
          v24 = v15 + 16;
          v25 = 3;
        }
        else
        {
          v29 = *(unsigned int *)(v15 + 24);
          v24 = *(_QWORD *)(v15 + 16);
          if ( !(_DWORD)v29 )
            goto LABEL_32;
          v25 = v29 - 1;
        }
        v26 = v25 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v27 = (__int64 *)(v24 + 16LL * v26);
        v28 = *v27;
        if ( v17 == *v27 )
          goto LABEL_23;
        v32 = 1;
        while ( v28 != -8 )
        {
          v33 = v32 + 1;
          v26 = v25 & (v32 + v26);
          v27 = (__int64 *)(v24 + 16LL * v26);
          v28 = *v27;
          if ( v17 == *v27 )
            goto LABEL_23;
          v32 = v33;
        }
        if ( (_BYTE)a4 )
        {
          v30 = 64;
          goto LABEL_33;
        }
        v29 = *(unsigned int *)(v15 + 24);
LABEL_32:
        v30 = 16 * v29;
LABEL_33:
        v27 = (__int64 *)(v24 + v30);
LABEL_23:
        a2 = 64;
        if ( !(_BYTE)a4 )
          a2 = 16LL * *(unsigned int *)(v15 + 24);
        if ( v27 == (__int64 *)(a2 + v24) )
        {
          v36 = v4;
          v31 = sub_14DB6D0(*v13, v39, v14, v15);
          a2 = v15;
          if ( v31 )
          {
            v34 = v36;
            v40 = v17;
            v41 = v31;
            v37 = v31;
            sub_14DB310((__int64)v42, v15, &v40, &v41);
            v4 = v34;
            v17 = v37;
          }
          else
          {
            v40 = v17;
            v41 = v17;
            sub_14DB310((__int64)v42, v15, &v40, &v41);
            v4 = v36;
          }
LABEL_9:
          v19 = (unsigned int)v44;
          if ( (unsigned int)v44 >= HIDWORD(v44) )
            goto LABEL_27;
          goto LABEL_10;
        }
        v17 = v27[1];
        v19 = (unsigned int)v44;
        if ( (unsigned int)v44 >= HIDWORD(v44) )
        {
LABEL_27:
          a2 = (__int64)v4;
          v35 = v4;
          sub_16CD150(&v43, v4, 0, 8);
          v19 = (unsigned int)v44;
          v4 = v35;
        }
LABEL_10:
        v13 += 3;
        *(_QWORD *)&v43[8 * v19] = v17;
        LODWORD(v44) = v44 + 1;
        if ( v16 == v13 )
        {
          a3 = v14;
          v6 = a1;
          v9 = v4;
          v7 = *(unsigned __int8 *)(a1 + 16);
          break;
        }
      }
    }
    if ( (_BYTE)v7 == 5 )
    {
      if ( (unsigned __int8)sub_1594520(v6, a2, v7, a4, v4) )
      {
        v20 = *(_QWORD **)v43;
        v21 = *((_QWORD *)v43 + 1);
        v22 = sub_1594720(v6);
        v8 = sub_14D7760(v22, v20, v21, v39, a3);
        if ( *(_BYTE *)(v8 + 16) == 5 )
          v8 = 0;
      }
      else
      {
        v8 = sub_14DCF80(v6, *(unsigned __int16 *)(v6 + 18), v43, (unsigned int)v44, v39, a3);
      }
    }
    else
    {
      v8 = sub_15A01B0(v43, (unsigned int)v44);
    }
    if ( v43 != v9 )
      _libc_free((unsigned __int64)v43);
  }
  return v8;
}
