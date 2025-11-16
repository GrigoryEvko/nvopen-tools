// Function: sub_14AEC90
// Address: 0x14aec90
//
__int64 __fastcall sub_14AEC90(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rcx
  _DWORD *v4; // rax
  unsigned int v5; // eax
  __int64 v6; // r14
  __int64 v7; // rsi
  __int64 result; // rax
  __int64 *v9; // rax
  __int64 *v10; // r15
  __int64 v11; // r15
  __int64 *v12; // rax
  __int64 *v13; // rax
  __int64 *v14; // rsi
  unsigned __int64 v15; // rdi
  unsigned int v16; // edx
  char v17; // cl
  _QWORD *v18; // rdx
  __int64 *v19; // r8
  __int64 *v20; // rdi
  __int64 *v21; // rdx
  __int64 *v22; // rcx
  unsigned int i; // [rsp+Ch] [rbp-144h]
  __int64 v24; // [rsp+10h] [rbp-140h]
  __int64 v25; // [rsp+18h] [rbp-138h]
  unsigned __int8 v26; // [rsp+18h] [rbp-138h]
  unsigned __int8 v27; // [rsp+18h] [rbp-138h]
  __int64 v28; // [rsp+20h] [rbp-130h] BYREF
  __int64 *v29; // [rsp+28h] [rbp-128h]
  __int64 *v30; // [rsp+30h] [rbp-120h]
  __int64 v31; // [rsp+38h] [rbp-118h]
  int v32; // [rsp+40h] [rbp-110h]
  _QWORD v33[5]; // [rsp+48h] [rbp-108h] BYREF
  __int64 v34; // [rsp+70h] [rbp-E0h] BYREF
  __int64 *v35; // [rsp+78h] [rbp-D8h]
  __int64 *v36; // [rsp+80h] [rbp-D0h]
  __int64 v37; // [rsp+88h] [rbp-C8h]
  int v38; // [rsp+90h] [rbp-C0h]
  _QWORD v39[23]; // [rsp+98h] [rbp-B8h] BYREF

  v2 = a1 + 24;
  v35 = v39;
  v3 = *(_QWORD *)(a1 + 40);
  v36 = v39;
  v29 = v33;
  v30 = v33;
  v37 = 0x100000010LL;
  v31 = 0x100000004LL;
  v24 = v3;
  v38 = 0;
  v39[0] = a1;
  v34 = 1;
  v32 = 0;
  v33[0] = v3;
  v28 = 1;
  v25 = v3 + 40;
  for ( i = 0; ; ++i )
  {
    v4 = (_DWORD *)sub_16D40F0(qword_4FBB370);
    v5 = v4 ? *v4 : LODWORD(qword_4FBB370[2]);
    if ( i >= v5 )
      break;
    while ( v25 != v2 )
    {
      if ( v2 )
      {
        v6 = v2 - 24;
        if ( a1 == v2 - 24 )
          goto LABEL_11;
      }
      else
      {
        v6 = 0;
      }
      v7 = sub_14AEA70(v6);
      if ( v7 && sub_14AEBB0((__int64)&v34, v7) )
      {
        v15 = (unsigned __int64)v30;
        v18 = v29;
        result = 1;
        goto LABEL_24;
      }
      result = sub_14AE440(v6);
      if ( !(_BYTE)result )
      {
        v15 = (unsigned __int64)v30;
        v18 = v29;
        goto LABEL_24;
      }
LABEL_11:
      v9 = v35;
      if ( v36 == v35 )
      {
        v10 = &v35[HIDWORD(v37)];
        if ( v35 == v10 )
        {
          v21 = v35;
        }
        else
        {
          do
          {
            if ( v6 == *v9 )
              break;
            ++v9;
          }
          while ( v10 != v9 );
          v21 = &v35[HIDWORD(v37)];
        }
        goto LABEL_36;
      }
      v10 = &v36[(unsigned int)v37];
      v9 = (__int64 *)sub_16CC9F0(&v34, v6);
      if ( v6 == *v9 )
      {
        if ( v36 == v35 )
          v21 = &v36[HIDWORD(v37)];
        else
          v21 = &v36[(unsigned int)v37];
LABEL_36:
        while ( v21 != v9 && (unsigned __int64)*v9 >= 0xFFFFFFFFFFFFFFFELL )
          ++v9;
        goto LABEL_15;
      }
      if ( v36 == v35 )
      {
        v9 = &v36[HIDWORD(v37)];
        v21 = v9;
        goto LABEL_36;
      }
      v9 = &v36[(unsigned int)v37];
LABEL_15:
      if ( v9 != v10 )
      {
LABEL_16:
        while ( 1 )
        {
          v6 = *(_QWORD *)(v6 + 8);
          if ( !v6 )
            break;
          while ( 1 )
          {
            v11 = sub_1648700(v6);
            if ( !(unsigned __int8)sub_14AEA40(v11) )
              break;
            v12 = v35;
            if ( v36 == v35 )
            {
              v19 = &v35[HIDWORD(v37)];
              if ( v35 != v19 )
              {
                v20 = 0;
                while ( v11 != *v12 )
                {
                  if ( *v12 == -2 )
                    v20 = v12;
                  if ( v19 == ++v12 )
                  {
                    if ( !v20 )
                      goto LABEL_50;
                    *v20 = v11;
                    --v38;
                    ++v34;
                    goto LABEL_16;
                  }
                }
                goto LABEL_16;
              }
LABEL_50:
              if ( HIDWORD(v37) < (unsigned int)v37 )
              {
                ++HIDWORD(v37);
                *v19 = v11;
                ++v34;
                goto LABEL_16;
              }
            }
            sub_16CCBA0(&v34, v11);
            v6 = *(_QWORD *)(v6 + 8);
            if ( !v6 )
              goto LABEL_20;
          }
        }
      }
LABEL_20:
      v2 = *(_QWORD *)(v2 + 8);
    }
    v24 = sub_157F1C0(v24);
    if ( !v24 )
      break;
    v13 = v29;
    if ( v30 != v29 )
      goto LABEL_23;
    v18 = (_QWORD *)HIDWORD(v31);
    v14 = &v29[HIDWORD(v31)];
    if ( v29 != v14 )
    {
      v22 = 0;
      do
      {
        v18 = (_QWORD *)*v13;
        if ( v24 == *v13 )
        {
          result = 0;
          goto LABEL_26;
        }
        if ( v18 == (_QWORD *)-2LL )
          v22 = v13;
        ++v13;
      }
      while ( v14 != v13 );
      if ( v22 )
      {
        *v22 = v24;
        --v32;
        ++v28;
        goto LABEL_64;
      }
    }
    if ( HIDWORD(v31) < (unsigned int)v31 )
    {
      ++HIDWORD(v31);
      *v14 = v24;
      ++v28;
    }
    else
    {
LABEL_23:
      v14 = (__int64 *)v24;
      sub_16CCBA0(&v28, v24);
      v15 = (unsigned __int64)v30;
      v17 = v16;
      result = v16;
      v18 = v29;
      if ( !v17 )
        goto LABEL_24;
    }
LABEL_64:
    v2 = sub_157ED20(v24, v14, v18) + 24;
    v25 = v24 + 40;
  }
  v15 = (unsigned __int64)v30;
  v18 = v29;
  result = 0;
LABEL_24:
  if ( v18 != (_QWORD *)v15 )
  {
    v26 = result;
    _libc_free(v15);
    result = v26;
  }
LABEL_26:
  if ( v36 != v35 )
  {
    v27 = result;
    _libc_free((unsigned __int64)v36);
    return v27;
  }
  return result;
}
