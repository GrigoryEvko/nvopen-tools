// Function: sub_D97520
// Address: 0xd97520
//
__int64 __fastcall sub_D97520(__int64 *a1, __int64 *a2, __int64 a3, _BYTE *a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r14
  __int64 *v7; // r12
  __int64 v9; // r15
  __int64 *v10; // rax
  __int64 v11; // rcx
  __int64 *v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // r12
  __int64 v15; // r14
  _BYTE *v16; // rax
  _BYTE *v17; // r15
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r9
  __int64 *v25; // r15
  __int64 *v26; // r14
  __int64 v27; // r8
  __int64 *v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // [rsp+18h] [rbp-118h]
  __int64 v33; // [rsp+18h] [rbp-118h]
  __int64 *v34; // [rsp+20h] [rbp-110h] BYREF
  __int64 v35; // [rsp+28h] [rbp-108h]
  _BYTE v36[48]; // [rsp+30h] [rbp-100h] BYREF
  __int64 v37; // [rsp+60h] [rbp-D0h] BYREF
  __int64 *v38; // [rsp+68h] [rbp-C8h]
  __int64 v39; // [rsp+70h] [rbp-C0h]
  int v40; // [rsp+78h] [rbp-B8h]
  char v41; // [rsp+7Ch] [rbp-B4h]
  char v42; // [rsp+80h] [rbp-B0h] BYREF

  v6 = &a2[a3];
  v38 = (__int64 *)&v42;
  v34 = (__int64 *)v36;
  *a4 = 1;
  v37 = 0;
  v39 = 16;
  v40 = 0;
  v41 = 1;
  v35 = 0x600000000LL;
  if ( a2 != v6 )
  {
    v7 = a2;
    a2 = (__int64 *)&v34;
    v9 = *v7;
LABEL_3:
    v10 = v38;
    v11 = HIDWORD(v39);
    v12 = &v38[HIDWORD(v39)];
    if ( v38 == v12 )
    {
LABEL_26:
      if ( HIDWORD(v39) >= (unsigned int)v39 )
        goto LABEL_9;
      v11 = (unsigned int)++HIDWORD(v39);
      *v12 = v9;
      ++v37;
LABEL_10:
      if ( (unsigned int)(HIDWORD(v39) - v40) <= 0x1E )
      {
        v19 = (unsigned int)v35;
        v11 = HIDWORD(v35);
        v20 = (unsigned int)v35 + 1LL;
        if ( v20 > HIDWORD(v35) )
        {
          a2 = (__int64 *)v36;
          sub_C8D5F0((__int64)&v34, v36, v20, 8u, a5, a6);
          v19 = (unsigned int)v35;
        }
        v12 = v34;
        v34[v19] = v9;
        LODWORD(v35) = v35 + 1;
        goto LABEL_7;
      }
      ++v7;
      *a4 = 0;
      if ( v6 != v7 )
        goto LABEL_8;
    }
    else
    {
      while ( v9 != *v10 )
      {
        if ( v12 == ++v10 )
          goto LABEL_26;
      }
LABEL_7:
      while ( v6 != ++v7 )
      {
LABEL_8:
        v9 = *v7;
        if ( v41 )
          goto LABEL_3;
LABEL_9:
        a2 = (__int64 *)v9;
        sub_C8CC70((__int64)&v37, v9, (__int64)v12, v11, a5, a6);
        if ( (_BYTE)v12 )
          goto LABEL_10;
      }
    }
    v13 = v35;
    if ( (_DWORD)v35 )
    {
      v14 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v15 = v34[v13 - 1];
          LODWORD(v35) = v13 - 1;
          a2 = (__int64 *)v15;
          v16 = sub_D974D0((__int64)a1, v15);
          v17 = v16;
          if ( v16 )
            break;
          v21 = (__int64 *)sub_D960E0(v15);
          v25 = &v21[v22];
          v26 = v21;
          if ( v25 != v21 )
          {
            v27 = *v21;
            if ( v41 )
            {
LABEL_33:
              v28 = v38;
              v23 = HIDWORD(v39);
              v22 = (__int64)&v38[HIDWORD(v39)];
              if ( v38 == (__int64 *)v22 )
                goto LABEL_43;
              while ( v27 != *v28 )
              {
                if ( (__int64 *)v22 == ++v28 )
                {
LABEL_43:
                  if ( HIDWORD(v39) < (unsigned int)v39 )
                  {
                    v23 = (unsigned int)++HIDWORD(v39);
                    *(_QWORD *)v22 = v27;
                    ++v37;
                    goto LABEL_40;
                  }
                  goto LABEL_39;
                }
              }
              goto LABEL_37;
            }
            while ( 1 )
            {
LABEL_39:
              a2 = (__int64 *)v27;
              v32 = v27;
              sub_C8CC70((__int64)&v37, v27, v22, v23, v27, v24);
              v27 = v32;
              if ( (_BYTE)v22 )
              {
LABEL_40:
                if ( (unsigned int)(HIDWORD(v39) - v40) <= 0x1E )
                {
                  v29 = (unsigned int)v35;
                  v23 = HIDWORD(v35);
                  v30 = (unsigned int)v35 + 1LL;
                  if ( v30 > HIDWORD(v35) )
                  {
                    a2 = (__int64 *)v36;
                    v33 = v27;
                    sub_C8D5F0((__int64)&v34, v36, v30, 8u, v27, v24);
                    v29 = (unsigned int)v35;
                    v27 = v33;
                  }
                  v22 = (__int64)v34;
                  ++v26;
                  v34[v29] = v27;
                  LODWORD(v35) = v35 + 1;
                  if ( v25 == v26 )
                    break;
                }
                else
                {
                  ++v26;
                  *a4 = 0;
                  if ( v25 == v26 )
                    break;
                }
              }
              else
              {
LABEL_37:
                if ( v25 == ++v26 )
                  break;
              }
              v27 = *v26;
              if ( v41 )
                goto LABEL_33;
            }
          }
LABEL_16:
          v13 = v35;
          if ( !(_DWORD)v35 )
            goto LABEL_20;
        }
        if ( v14 )
        {
          a2 = (__int64 *)v14;
          if ( (unsigned __int8)sub_B19DB0(a1[5], v14, (__int64)v16) )
            v14 = (__int64)v17;
          goto LABEL_16;
        }
        v13 = v35;
        v14 = (__int64)v17;
        if ( !(_DWORD)v35 )
        {
LABEL_20:
          if ( v14 )
            goto LABEL_21;
          break;
        }
      }
    }
  }
  v31 = *(_QWORD *)(*a1 + 80);
  if ( !v31 )
    BUG();
  v14 = *(_QWORD *)(v31 + 32);
  if ( v14 )
    v14 -= 24;
LABEL_21:
  if ( v34 != (__int64 *)v36 )
    _libc_free(v34, a2);
  if ( !v41 )
    _libc_free(v38, a2);
  return v14;
}
