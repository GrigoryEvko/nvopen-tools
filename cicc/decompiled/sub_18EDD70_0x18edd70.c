// Function: sub_18EDD70
// Address: 0x18edd70
//
__int64 __fastcall sub_18EDD70(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r15
  __int64 v12; // r8
  __int64 i; // rcx
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // r12
  __int64 v19; // rax
  __int64 v20; // r14
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 j; // r15
  int *v24; // r15
  int *v25; // r13
  int *v26; // rax
  unsigned __int8 v27; // r13
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  int *v30; // rcx
  __int64 v31; // rdi
  __int64 v32; // rsi
  __int64 v33; // rcx
  int *v34; // rdi
  __int64 v35; // rax
  bool v36; // al
  _BYTE *v37; // [rsp+10h] [rbp-90h]
  unsigned __int8 v38; // [rsp+1Fh] [rbp-81h]
  __int64 v39; // [rsp+20h] [rbp-80h]
  __int64 v40; // [rsp+20h] [rbp-80h]
  __int64 v41; // [rsp+28h] [rbp-78h]
  _QWORD *v42; // [rsp+38h] [rbp-68h] BYREF
  __int64 v43; // [rsp+40h] [rbp-60h] BYREF
  int v44; // [rsp+48h] [rbp-58h] BYREF
  int *v45; // [rsp+50h] [rbp-50h]
  int *v46; // [rsp+58h] [rbp-48h]
  int *v47; // [rsp+60h] [rbp-40h]
  __int64 v48; // [rsp+68h] [rbp-38h]

  v38 = sub_1636880(a1, a2);
  if ( v38 )
  {
    return 0;
  }
  else
  {
    v11 = *(_QWORD *)(a2 + 80);
    v12 = a2 + 72;
    v44 = 0;
    v45 = 0;
    v46 = &v44;
    v47 = &v44;
    v48 = 0;
    if ( a2 + 72 == v11 )
    {
      i = 0;
    }
    else
    {
      if ( !v11 )
        BUG();
      while ( 1 )
      {
        i = *(_QWORD *)(v11 + 24);
        if ( i != v11 + 16 )
          break;
        v11 = *(_QWORD *)(v11 + 8);
        if ( v12 == v11 )
          goto LABEL_10;
        if ( !v11 )
          BUG();
      }
    }
    if ( v12 != v11 )
    {
      do
      {
        v28 = i - 24;
        if ( !i )
          v28 = 0;
        v40 = v12;
        v41 = i;
        v42 = (_QWORD *)v28;
        sub_18EDC50(&v43, (unsigned __int64 *)&v42);
        v12 = v40;
        for ( i = *(_QWORD *)(v41 + 8); ; i = *(_QWORD *)(v11 + 24) )
        {
          v29 = v11 - 24;
          if ( !v11 )
            v29 = 0;
          if ( i != v29 + 40 )
            break;
          v11 = *(_QWORD *)(v11 + 8);
          if ( v40 == v11 )
            goto LABEL_10;
          if ( !v11 )
            BUG();
        }
      }
      while ( v40 != v11 );
    }
LABEL_10:
    v37 = (_BYTE *)sub_1632FA0(*(_QWORD *)(a2 + 40));
    v14 = *(__int64 **)(a1 + 8);
    v15 = *v14;
    v16 = v14[1];
    if ( v15 == v16 )
LABEL_64:
      BUG();
    while ( *(_UNKNOWN **)v15 != &unk_4F9B6E8 )
    {
      v15 += 16;
      if ( v16 == v15 )
        goto LABEL_64;
    }
    v39 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(
            *(_QWORD *)(v15 + 8),
            &unk_4F9B6E8)
        + 360;
LABEL_15:
    v17 = v48;
    while ( v17 )
    {
      v18 = (__int64 *)*((_QWORD *)v46 + 4);
      v19 = sub_220F330(v46, &v44);
      j_j___libc_free_0(v19, 40);
      v17 = --v48;
      if ( v18[1] )
      {
        v20 = sub_14DD210(v18, v37, v39);
        if ( !v20 )
          goto LABEL_15;
        for ( j = v18[1]; j; j = *(_QWORD *)(j + 8) )
        {
          v42 = sub_1648700(j);
          sub_18EDC50(&v43, (unsigned __int64 *)&v42);
        }
        v24 = &v44;
        sub_164D160((__int64)v18, v20, a3, a4, a5, a6, v21, v22, a9, a10);
        v25 = v45;
        if ( v45 )
        {
          while ( 1 )
          {
            while ( *((_QWORD *)v25 + 4) < (unsigned __int64)v18 )
            {
              v25 = (int *)*((_QWORD *)v25 + 3);
              if ( !v25 )
                goto LABEL_27;
            }
            v26 = (int *)*((_QWORD *)v25 + 2);
            if ( *((_QWORD *)v25 + 4) <= (unsigned __int64)v18 )
              break;
            v24 = v25;
            v25 = (int *)*((_QWORD *)v25 + 2);
            if ( !v26 )
            {
LABEL_27:
              v36 = v24 == &v44;
              goto LABEL_28;
            }
          }
          v30 = (int *)*((_QWORD *)v25 + 3);
          if ( v30 )
          {
            do
            {
              while ( 1 )
              {
                v31 = *((_QWORD *)v30 + 2);
                v32 = *((_QWORD *)v30 + 3);
                if ( *((_QWORD *)v30 + 4) > (unsigned __int64)v18 )
                  break;
                v30 = (int *)*((_QWORD *)v30 + 3);
                if ( !v32 )
                  goto LABEL_52;
              }
              v24 = v30;
              v30 = (int *)*((_QWORD *)v30 + 2);
            }
            while ( v31 );
          }
LABEL_52:
          while ( v26 )
          {
            while ( 1 )
            {
              v33 = *((_QWORD *)v26 + 3);
              if ( *((_QWORD *)v26 + 4) >= (unsigned __int64)v18 )
                break;
              v26 = (int *)*((_QWORD *)v26 + 3);
              if ( !v33 )
                goto LABEL_55;
            }
            v25 = v26;
            v26 = (int *)*((_QWORD *)v26 + 2);
          }
LABEL_55:
          if ( v46 == v25 && v24 == &v44 )
          {
LABEL_30:
            sub_18ED7D0((__int64)v45);
            v45 = 0;
            v46 = &v44;
            v47 = &v44;
            v48 = 0;
          }
          else
          {
            for ( ; v25 != v24; --v48 )
            {
              v34 = v25;
              v25 = (int *)sub_220EF30(v25);
              v35 = sub_220F330(v34, &v44);
              j_j___libc_free_0(v35, 40);
            }
          }
        }
        else
        {
          v36 = 1;
LABEL_28:
          if ( v46 == v24 && v36 )
            goto LABEL_30;
        }
        v38 = 1;
        v27 = sub_1AE9990(v18, v39);
        if ( v27 )
        {
          sub_15F20C0(v18);
          v38 = v27;
        }
        goto LABEL_15;
      }
    }
    sub_18ED7D0((__int64)v45);
  }
  return v38;
}
