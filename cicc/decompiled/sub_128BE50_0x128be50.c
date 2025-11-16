// Function: sub_128BE50
// Address: 0x128be50
//
__int64 __fastcall sub_128BE50(__int64 a1, _BYTE *a2, __int64 *a3, __int64 a4, char a5)
{
  __int64 v8; // r15
  unsigned int v9; // r13d
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // r15
  __int64 v13; // rax
  _BYTE *v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 *v18; // rbx
  _QWORD *v19; // r13
  __int64 v20; // rdx
  __int64 *v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // rax
  __int64 *v25; // r9
  __int64 *v26; // r10
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rdi
  unsigned __int64 *v35; // r12
  __int64 v36; // rax
  unsigned __int64 v37; // rcx
  __int64 v38; // rsi
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 *v41; // r10
  __int64 v42; // r9
  __int64 v43; // rax
  __int64 *v44; // rdx
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 *v48; // rdx
  __int64 v49; // rsi
  __int64 *v50; // [rsp+0h] [rbp-A0h]
  __int64 *v51; // [rsp+8h] [rbp-98h]
  __int64 *v52; // [rsp+8h] [rbp-98h]
  unsigned int v53; // [rsp+10h] [rbp-90h]
  __int64 *v54; // [rsp+10h] [rbp-90h]
  __int64 *v55; // [rsp+10h] [rbp-90h]
  __int64 v56; // [rsp+10h] [rbp-90h]
  __int64 *v57; // [rsp+10h] [rbp-90h]
  __int64 *v58; // [rsp+10h] [rbp-90h]
  __int64 *v59; // [rsp+18h] [rbp-88h] BYREF
  __int64 v60; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v61[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v62; // [rsp+40h] [rbp-60h]
  _QWORD v63[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v64; // [rsp+60h] [rbp-40h]

  v59 = a3;
  v8 = *(_QWORD *)a2;
  if ( !(unsigned __int8)sub_127B3A0(a4) )
  {
    v53 = *(_DWORD *)(*v59 + 8) >> 8;
    v23 = sub_127B390();
    if ( v23 > v53 )
    {
      v24 = sub_1644900(*(_QWORD *)(a1 + 16), v23);
      v25 = v59;
      v62 = 259;
      v26 = *(__int64 **)(a1 + 8);
      v61[0] = "idx.ext";
      if ( v24 != *v59 )
      {
        if ( *((_BYTE *)v59 + 16) > 0x10u )
        {
          v64 = 257;
          v55 = v26;
          v40 = sub_15FDBD0(37, v59, v24, v63, 0);
          v41 = v55;
          v42 = v40;
          v43 = v55[1];
          if ( v43 )
          {
            v44 = (__int64 *)v55[2];
            v50 = v55;
            v56 = v42;
            v51 = v44;
            sub_157E9D0(v43 + 40, v42);
            v42 = v56;
            v41 = v50;
            v45 = *v51;
            v46 = *(_QWORD *)(v56 + 24);
            *(_QWORD *)(v56 + 32) = v51;
            v45 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v56 + 24) = v45 | v46 & 7;
            *(_QWORD *)(v45 + 8) = v56 + 24;
            *v51 = *v51 & 7 | (v56 + 24);
          }
          v52 = v41;
          v57 = (__int64 *)v42;
          sub_164B780(v42, v61);
          v25 = v57;
          v47 = *v52;
          if ( *v52 )
          {
            v60 = *v52;
            sub_1623A60(&v60, v47, 2);
            v25 = v57;
            v48 = v57 + 6;
            if ( v57[6] )
            {
              sub_161E7C0(v57 + 6);
              v25 = v57;
              v48 = v57 + 6;
            }
            v49 = v60;
            v25[6] = v60;
            if ( v49 )
            {
              v58 = v25;
              sub_1623210(&v60, v49, v48);
              v25 = v58;
            }
          }
        }
        else
        {
          v25 = (__int64 *)sub_15A46C0(37, v59, v24, 0);
        }
      }
      v59 = v25;
    }
  }
  if ( a5 || *(_BYTE *)(*(_QWORD *)(v8 + 24) + 8LL) == 12 )
  {
    v9 = *(_DWORD *)(v8 + 8);
    v10 = sub_1643330(*(_QWORD *)(a1 + 16));
    v11 = sub_1646BA0(v10, v9 >> 8);
    v12 = *(__int64 **)(a1 + 8);
    v62 = 257;
    if ( v11 == *(_QWORD *)a2 )
    {
      v14 = a2;
    }
    else if ( a2[16] > 0x10u )
    {
      v64 = 257;
      v27 = sub_15FDBD0(47, a2, v11, v63, 0);
      v28 = v12[1];
      v14 = (_BYTE *)v27;
      if ( v28 )
      {
        v54 = (__int64 *)v12[2];
        sub_157E9D0(v28 + 40, v27);
        v29 = *v54;
        v30 = *((_QWORD *)v14 + 3) & 7LL;
        *((_QWORD *)v14 + 4) = v54;
        v29 &= 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)v14 + 3) = v29 | v30;
        *(_QWORD *)(v29 + 8) = v14 + 24;
        *v54 = *v54 & 7 | (unsigned __int64)(v14 + 24);
      }
      sub_164B780(v14, v61);
      v31 = *v12;
      if ( *v12 )
      {
        v60 = *v12;
        sub_1623A60(&v60, v31, 2);
        if ( *((_QWORD *)v14 + 6) )
          sub_161E7C0(v14 + 48);
        v32 = v60;
        *((_QWORD *)v14 + 6) = v60;
        if ( v32 )
          sub_1623210(&v60, v32, v14 + 48);
      }
      v12 = *(__int64 **)(a1 + 8);
    }
    else
    {
      v13 = sub_15A46C0(47, a2, v11, 0);
      v12 = *(__int64 **)(a1 + 8);
      v14 = (_BYTE *)v13;
    }
    v15 = *(_QWORD *)(a1 + 16);
    v63[0] = "add.ptr";
    v64 = 259;
    v16 = sub_1643330(v15);
    v17 = sub_12815B0(v12, v16, v14, (__int64)v59, (__int64)v63);
    v18 = *(__int64 **)(a1 + 8);
    v62 = 257;
    v19 = (_QWORD *)v17;
    v20 = *(_QWORD *)a2;
    if ( *(_QWORD *)a2 != *(_QWORD *)v17 )
    {
      if ( *(_BYTE *)(v17 + 16) > 0x10u )
      {
        v64 = 257;
        v33 = sub_15FDBD0(47, v17, v20, v63, 0);
        v34 = v18[1];
        v19 = (_QWORD *)v33;
        if ( v34 )
        {
          v35 = (unsigned __int64 *)v18[2];
          sub_157E9D0(v34 + 40, v33);
          v36 = v19[3];
          v37 = *v35;
          v19[4] = v35;
          v37 &= 0xFFFFFFFFFFFFFFF8LL;
          v19[3] = v37 | v36 & 7;
          *(_QWORD *)(v37 + 8) = v19 + 3;
          *v35 = *v35 & 7 | (unsigned __int64)(v19 + 3);
        }
        sub_164B780(v19, v61);
        v38 = *v18;
        if ( *v18 )
        {
          v60 = *v18;
          sub_1623A60(&v60, v38, 2);
          if ( v19[6] )
            sub_161E7C0(v19 + 6);
          v39 = v60;
          v19[6] = v60;
          if ( v39 )
            sub_1623210(&v60, v39, v19 + 6);
        }
      }
      else
      {
        return sub_15A46C0(47, v17, v20, 0);
      }
    }
  }
  else
  {
    v22 = *(__int64 **)(a1 + 8);
    v63[0] = "add.ptr";
    v64 = 259;
    return sub_128B460(v22, 0, a2, &v59, 1, (__int64)v63);
  }
  return (__int64)v19;
}
