// Function: sub_24677C0
// Address: 0x24677c0
//
__int64 __fastcall sub_24677C0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        unsigned __int8 a7)
{
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // eax
  unsigned int v12; // ebx
  char v13; // r12
  __int64 result; // rax
  __int64 v15; // r13
  char v16; // r15
  _QWORD *v17; // rax
  __int64 v18; // r9
  __int64 v19; // r12
  unsigned int *v20; // r13
  __int64 v21; // rbx
  __int64 v22; // rdx
  unsigned int v23; // esi
  _BYTE *v24; // r12
  __int64 v25; // rax
  _BYTE *v26; // rax
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // rdi
  _BYTE *v31; // rdx
  __int64 v32; // rax
  unsigned __int16 v33; // cx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  _BYTE *v37; // r9
  char v38; // al
  __int64 v39; // rdx
  unsigned int v40; // r13d
  __int64 v41; // rax
  unsigned __int16 v42; // bx
  int v43; // r13d
  __int64 v44; // rax
  unsigned int *v45; // rbx
  __int64 v46; // rdx
  unsigned int v47; // esi
  unsigned int v48; // [rsp+Ch] [rbp-124h]
  __int64 v49; // [rsp+10h] [rbp-120h]
  __int64 v50; // [rsp+18h] [rbp-118h]
  __int64 v51; // [rsp+18h] [rbp-118h]
  __int64 v52; // [rsp+18h] [rbp-118h]
  _BYTE *v53; // [rsp+20h] [rbp-110h]
  _BYTE *v54; // [rsp+20h] [rbp-110h]
  unsigned int v56; // [rsp+38h] [rbp-F8h]
  unsigned __int8 v57; // [rsp+3Ch] [rbp-F4h]
  int v58; // [rsp+3Ch] [rbp-F4h]
  unsigned int v61; // [rsp+58h] [rbp-D8h]
  unsigned int v62; // [rsp+58h] [rbp-D8h]
  char v65[32]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v66; // [rsp+90h] [rbp-A0h]
  __int64 v67; // [rsp+A0h] [rbp-90h] BYREF
  _BYTE *v68; // [rsp+A8h] [rbp-88h] BYREF
  __int16 v69; // [rsp+C0h] [rbp-70h]
  unsigned __int64 v70; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v71; // [rsp+D8h] [rbp-58h]
  __int16 v72; // [rsp+F0h] [rbp-40h]

  v8 = sub_B2BEC0(*a1);
  v57 = sub_AE5020(v8, *(_QWORD *)(a1[1] + 80));
  v9 = sub_9208B0(v8, *(_QWORD *)(a1[1] + 80));
  v71 = v10;
  v70 = (unsigned __int64)(v9 + 7) >> 3;
  v11 = sub_CA1930(&v70);
  v61 = v11;
  if ( a6 )
  {
    v24 = (_BYTE *)sub_B33F60(a2, *(_QWORD *)(a1[1] + 80), a5, a6);
    v25 = a1[1];
    v72 = 257;
    v26 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v25 + 80), 3, 0);
    v27 = sub_929C50((unsigned int **)a2, v24, v26, (__int64)&v70, 0, 0);
    v28 = a1[1];
    v69 = 257;
    v29 = sub_AD64C0(*(_QWORD *)(v28 + 80), 4, 0);
    v30 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a2 + 80) + 24LL))(
            *(_QWORD *)(a2 + 80),
            19,
            v27,
            v29,
            0);
    if ( !v30 )
    {
      v72 = 257;
      v44 = sub_B504D0(19, v27, v29, (__int64)&v70, 0, 0);
      v30 = sub_1157250((__int64 *)a2, v44, (__int64)&v67);
    }
    v67 = sub_F369B0(v30, *(__int64 **)(a2 + 56), *(unsigned __int16 *)(a2 + 64));
    v68 = v31;
    sub_D5F1F0(a2, v67);
    v72 = 257;
    v32 = sub_921130((unsigned int **)a2, *(_QWORD *)(a1[1] + 88), a4, &v68, 1, (__int64)&v70, 0);
    LOBYTE(v33) = byte_4FE8EA9;
    HIBYTE(v33) = 1;
    return sub_2463EC0((__int64 *)a2, a3, v32, v33, 0);
  }
  else
  {
    v12 = v11;
    v13 = a7;
    if ( v11 <= 4 || a7 < v57 )
      goto LABEL_4;
    v34 = sub_B2BEC0(*a1);
    v70 = sub_9C6480(v34, *(_QWORD *)(a1[1] + 80));
    v71 = v35;
    v50 = a3;
    if ( (unsigned int)sub_CA1930(&v70) != 4 )
    {
      v72 = 257;
      v36 = sub_921630((unsigned int **)a2, a3, *(_QWORD *)(a1[1] + 80), 0, (__int64)&v70);
      v69 = 257;
      v66 = 257;
      v53 = (_BYTE *)v36;
      v51 = sub_AD64C0(*(_QWORD *)(v36 + 8), 32, 0);
      v37 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, __int64, _QWORD, _QWORD))(**(_QWORD **)(a2 + 80)
                                                                                                  + 32LL))(
                       *(_QWORD *)(a2 + 80),
                       25,
                       v53,
                       v51,
                       0,
                       0);
      if ( !v37 )
      {
        v72 = 257;
        v49 = sub_B504D0(25, (__int64)v53, v51, (__int64)&v70, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
          *(_QWORD *)(a2 + 88),
          v49,
          v65,
          *(_QWORD *)(a2 + 56),
          *(_QWORD *)(a2 + 64));
        v37 = (_BYTE *)v49;
        v52 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v52 )
        {
          v48 = v12;
          v45 = *(unsigned int **)a2;
          do
          {
            v46 = *((_QWORD *)v45 + 1);
            v47 = *v45;
            v45 += 4;
            sub_B99FD0(v49, v47, v46);
          }
          while ( (unsigned int *)v52 != v45 );
          v37 = (_BYTE *)v49;
          v12 = v48;
          v13 = a7;
        }
      }
      v50 = sub_A82480((unsigned int **)a2, v53, v37, (__int64)&v67);
    }
    v72 = 257;
    v54 = sub_94BCF0((unsigned int **)a2, a4, *(_QWORD *)(a1[1] + 96), (__int64)&v70);
    v56 = (unsigned int)a5 / v12;
    if ( v61 > (unsigned int)a5 )
    {
LABEL_4:
      v62 = 0;
    }
    else
    {
      v38 = v13;
      v39 = (__int64)v54;
      v40 = 0;
      while ( 1 )
      {
        LOBYTE(v42) = v38;
        HIBYTE(v42) = 1;
        ++v40;
        sub_2463EC0((__int64 *)a2, v50, v39, v42, 0);
        if ( v40 >= v56 )
          break;
        v41 = a1[1];
        v72 = 257;
        v39 = sub_94B060((unsigned int **)a2, *(_QWORD *)(v41 + 80), (__int64)v54, v40, (__int64)&v70);
        v38 = v57;
      }
      v43 = 1;
      if ( v61 <= (unsigned int)a5 )
        v43 = v56;
      v13 = v57;
      v62 = (v61 >> 2) * v43;
    }
    result = (unsigned int)(a5 + 3) >> 2;
    v58 = result;
    if ( v62 < (unsigned int)result )
    {
      do
      {
        v15 = a4;
        if ( v62 )
        {
          v72 = 257;
          v15 = sub_94B060((unsigned int **)a2, *(_QWORD *)(a1[1] + 88), a4, v62, (__int64)&v70);
        }
        v16 = v13;
        v72 = 257;
        v17 = sub_BD2C40(80, unk_3F10A10);
        v19 = (__int64)v17;
        if ( v17 )
          sub_B4D3C0((__int64)v17, a3, v15, 0, v16, v18, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
          *(_QWORD *)(a2 + 88),
          v19,
          &v70,
          *(_QWORD *)(a2 + 56),
          *(_QWORD *)(a2 + 64));
        v20 = *(unsigned int **)a2;
        v21 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v21 )
        {
          do
          {
            v22 = *((_QWORD *)v20 + 1);
            v23 = *v20;
            v20 += 4;
            sub_B99FD0(v19, v23, v22);
          }
          while ( (unsigned int *)v21 != v20 );
        }
        ++v62;
        v13 = byte_4FE8EA9;
        result = v62;
      }
      while ( v62 != v58 );
    }
  }
  return result;
}
