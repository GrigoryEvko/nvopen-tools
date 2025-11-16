// Function: sub_14ADFA0
// Address: 0x14adfa0
//
__int64 __fastcall sub_14ADFA0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 ***v5; // r12
  __int64 v7; // rax
  __int64 *v8; // r13
  __int64 v9; // rax
  __int64 *v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 *v14; // r15
  __int64 **v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // r15
  __int64 **v20; // r14
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // r15
  __int64 **v25; // r14
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 *v29; // r15
  __int64 **v30; // r14
  __int64 i; // r14
  __int64 v32; // r15
  __int64 v33; // rax
  __int64 *v34; // [rsp+8h] [rbp-A8h]
  __int64 *v35; // [rsp+18h] [rbp-98h]
  __int64 **v36; // [rsp+20h] [rbp-90h] BYREF
  __int64 v37; // [rsp+28h] [rbp-88h]
  __int64 v38; // [rsp+30h] [rbp-80h] BYREF
  __int64 v39; // [rsp+38h] [rbp-78h]
  __int64 *v40; // [rsp+40h] [rbp-70h] BYREF
  __int64 v41; // [rsp+48h] [rbp-68h]
  _BYTE v42[16]; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v43; // [rsp+60h] [rbp-50h] BYREF
  __int64 v44; // [rsp+68h] [rbp-48h]
  _BYTE v45[64]; // [rsp+70h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  v40 = (__int64 *)v42;
  v41 = 0x200000000LL;
  v43 = v45;
  v44 = 0x200000000LL;
  if ( !v2 )
    goto LABEL_3;
  do
  {
    v3 = sub_1648700(v2);
    v4 = v3;
    if ( *(_BYTE *)(v3 + 16) != 86 )
      goto LABEL_3;
    if ( **(_DWORD **)(v3 + 56) )
    {
      for ( i = *(_QWORD *)(v3 + 8); i; i = *(_QWORD *)(i + 8) )
      {
        v32 = sub_1648700(i);
        if ( *(_BYTE *)(v32 + 16) == 26 )
        {
          v33 = (unsigned int)v41;
          if ( (unsigned int)v41 >= HIDWORD(v41) )
          {
            sub_16CD150(&v40, v42, 0, 8);
            v33 = (unsigned int)v41;
          }
          v40[v33] = v32;
          LODWORD(v41) = v41 + 1;
        }
      }
    }
    else
    {
      v7 = (unsigned int)v44;
      if ( (unsigned int)v44 >= HIDWORD(v44) )
      {
        sub_16CD150(&v43, v45, 0, 8);
        v7 = (unsigned int)v44;
      }
      *(_QWORD *)&v43[8 * v7] = v4;
      LODWORD(v44) = v44 + 1;
    }
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( v2 );
  v8 = v40;
  v37 = a2;
  v36 = (__int64 **)&v43;
  v9 = 8LL * (unsigned int)v41;
  v10 = &v40[(unsigned __int64)v9 / 8];
  v11 = v9 >> 3;
  v12 = v9 >> 5;
  v34 = v10;
  if ( v12 )
  {
    v35 = &v40[4 * v12];
    while ( 1 )
    {
      v13 = *(_QWORD *)(*v8 + 40);
      v39 = *(_QWORD *)(*v8 - 48);
      v38 = v13;
      if ( (unsigned __int8)sub_15CC350(&v38) )
        break;
LABEL_23:
      v16 = v8[1];
      v17 = *(_QWORD *)(v16 - 48);
      v18 = *(_QWORD *)(v16 + 40);
      v39 = v17;
      v38 = v18;
      if ( (unsigned __int8)sub_15CC350(&v38) )
      {
        v5 = (__int64 ***)*v36;
        v19 = &(*v36)[*((unsigned int *)v36 + 2)];
        if ( *v36 != v19 )
        {
          do
          {
            v20 = *v5;
            if ( !(unsigned __int8)sub_15CCD40(v37, &v38, (*v5)[5]) )
            {
              while ( 1 )
              {
                v20 = (__int64 **)v20[1];
                if ( !v20 )
                  break;
                if ( !(unsigned __int8)sub_15CCFD0(v37, &v38, v20) )
                  goto LABEL_30;
              }
            }
            ++v5;
          }
          while ( v19 != (__int64 *)v5 );
        }
        LOBYTE(v5) = v34 != v8 + 1;
        goto LABEL_4;
      }
LABEL_30:
      v21 = v8[2];
      v22 = *(_QWORD *)(v21 - 48);
      v23 = *(_QWORD *)(v21 + 40);
      v39 = v22;
      v38 = v23;
      if ( (unsigned __int8)sub_15CC350(&v38) )
      {
        v5 = (__int64 ***)*v36;
        v24 = &(*v36)[*((unsigned int *)v36 + 2)];
        if ( *v36 != v24 )
        {
          do
          {
            v25 = *v5;
            if ( !(unsigned __int8)sub_15CCD40(v37, &v38, (*v5)[5]) )
            {
              while ( 1 )
              {
                v25 = (__int64 **)v25[1];
                if ( !v25 )
                  break;
                if ( !(unsigned __int8)sub_15CCFD0(v37, &v38, v25) )
                  goto LABEL_37;
              }
            }
            ++v5;
          }
          while ( v24 != (__int64 *)v5 );
        }
        LOBYTE(v5) = v34 != v8 + 2;
        goto LABEL_4;
      }
LABEL_37:
      v26 = v8[3];
      v27 = *(_QWORD *)(v26 - 48);
      v28 = *(_QWORD *)(v26 + 40);
      v39 = v27;
      v38 = v28;
      if ( (unsigned __int8)sub_15CC350(&v38) )
      {
        v5 = (__int64 ***)*v36;
        v29 = &(*v36)[*((unsigned int *)v36 + 2)];
        if ( *v36 != v29 )
        {
          do
          {
            v30 = *v5;
            if ( !(unsigned __int8)sub_15CCD40(v37, &v38, (*v5)[5]) )
            {
              while ( 1 )
              {
                v30 = (__int64 **)v30[1];
                if ( !v30 )
                  break;
                if ( !(unsigned __int8)sub_15CCFD0(v37, &v38, v30) )
                  goto LABEL_44;
              }
            }
            ++v5;
          }
          while ( v29 != (__int64 *)v5 );
        }
        LOBYTE(v5) = v34 != v8 + 3;
        goto LABEL_4;
      }
LABEL_44:
      v8 += 4;
      if ( v35 == v8 )
      {
        v11 = v34 - v8;
        goto LABEL_46;
      }
    }
    v5 = (__int64 ***)*v36;
    v14 = &(*v36)[*((unsigned int *)v36 + 2)];
    if ( *v36 != v14 )
    {
      do
      {
        v15 = *v5;
        if ( !(unsigned __int8)sub_15CCD40(v37, &v38, (*v5)[5]) )
        {
          while ( 1 )
          {
            v15 = (__int64 **)v15[1];
            if ( !v15 )
              break;
            if ( !(unsigned __int8)sub_15CCFD0(v37, &v38, v15) )
              goto LABEL_23;
          }
        }
        ++v5;
      }
      while ( v14 != (__int64 *)v5 );
    }
    goto LABEL_20;
  }
LABEL_46:
  v5 = &v36;
  if ( v11 != 2 )
  {
    if ( v11 != 3 )
    {
      if ( v11 == 1 )
        goto LABEL_49;
LABEL_3:
      LODWORD(v5) = 0;
      goto LABEL_4;
    }
    v5 = &v36;
    if ( (unsigned __int8)sub_14A8CD0(&v36, *v8) )
    {
LABEL_20:
      LOBYTE(v5) = v34 != v8;
      goto LABEL_4;
    }
    ++v8;
  }
  if ( (unsigned __int8)sub_14A8CD0(&v36, *v8) )
    goto LABEL_20;
  ++v8;
LABEL_49:
  LODWORD(v5) = sub_14A8CD0(&v36, *v8);
  if ( (_BYTE)v5 )
    goto LABEL_20;
LABEL_4:
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
  if ( v40 != (__int64 *)v42 )
    _libc_free((unsigned __int64)v40);
  return (unsigned int)v5;
}
