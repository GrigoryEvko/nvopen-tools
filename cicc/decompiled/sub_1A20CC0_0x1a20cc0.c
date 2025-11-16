// Function: sub_1A20CC0
// Address: 0x1a20cc0
//
__int64 __fastcall sub_1A20CC0(__int64 a1, unsigned __int64 *a2)
{
  __int64 v2; // r14
  _QWORD *v4; // r15
  _BYTE *v5; // rbx
  int v6; // r8d
  int v7; // r9d
  char v8; // dl
  _QWORD *v9; // r13
  _QWORD *v10; // rsi
  _QWORD *v11; // rcx
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rbx
  int v17; // eax
  __int64 v18; // rsi
  unsigned __int64 v19; // rax
  _BYTE *v20; // rdi
  __int64 **v21; // rdx
  __int64 v22; // r13
  __int64 *v23; // rcx
  char v24; // al
  __int64 *v25; // rax
  __int64 v26; // r14
  int v27; // r8d
  int v28; // r9d
  char v29; // dl
  _QWORD *v30; // r15
  _QWORD *v31; // rax
  _QWORD *v32; // rsi
  _QWORD *v33; // rcx
  __int64 v34; // rax
  __int64 *v35; // rax
  __int64 v37; // [rsp+20h] [rbp-D0h] BYREF
  _BYTE *v38; // [rsp+28h] [rbp-C8h]
  _BYTE *v39; // [rsp+30h] [rbp-C0h]
  __int64 v40; // [rsp+38h] [rbp-B8h]
  int v41; // [rsp+40h] [rbp-B0h]
  _BYTE v42[40]; // [rsp+48h] [rbp-A8h] BYREF
  _BYTE *v43; // [rsp+70h] [rbp-80h] BYREF
  __int64 v44; // [rsp+78h] [rbp-78h]
  _BYTE v45[112]; // [rsp+80h] [rbp-70h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  v43 = v45;
  v37 = 0;
  v38 = v42;
  v39 = v42;
  v40 = 4;
  v41 = 0;
  v44 = 0x400000000LL;
  if ( v2 )
  {
    v4 = v42;
    v5 = v42;
    while ( 1 )
    {
      v9 = sub_1648700(v2);
      if ( v5 == (_BYTE *)v4 )
      {
        v10 = &v4[HIDWORD(v40)];
        if ( v10 != v4 )
        {
          v11 = 0;
          while ( v9 != (_QWORD *)*v4 )
          {
            if ( *v4 == -2 )
              v11 = v4;
            if ( v10 == ++v4 )
            {
              if ( !v11 )
                goto LABEL_51;
              *v11 = v9;
              --v41;
              ++v37;
              goto LABEL_15;
            }
          }
          goto LABEL_4;
        }
LABEL_51:
        if ( HIDWORD(v40) < (unsigned int)v40 )
          break;
      }
      sub_16CCBA0((__int64)&v37, (__int64)v9);
      if ( v8 )
      {
LABEL_15:
        v12 = (unsigned int)v44;
        if ( (unsigned int)v44 < HIDWORD(v44) )
          goto LABEL_16;
        goto LABEL_53;
      }
LABEL_4:
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        goto LABEL_17;
LABEL_5:
      v5 = v39;
      v4 = v38;
    }
    ++HIDWORD(v40);
    *v10 = v9;
    v12 = (unsigned int)v44;
    ++v37;
    if ( (unsigned int)v44 < HIDWORD(v44) )
      goto LABEL_16;
LABEL_53:
    sub_16CD150((__int64)&v43, v45, 0, 16, v6, v7);
    v12 = (unsigned int)v44;
LABEL_16:
    v13 = (__int64 *)&v43[16 * v12];
    v13[1] = (__int64)v9;
    *v13 = a1;
    LODWORD(v44) = v44 + 1;
    v2 = *(_QWORD *)(v2 + 8);
    if ( v2 )
      goto LABEL_5;
LABEL_17:
    if ( (_DWORD)v44 )
    {
      v14 = sub_15F2050(a1);
      v15 = sub_1632FA0(v14);
      *a2 = 0;
      v16 = v15;
      v17 = v44;
      do
      {
        v20 = v43;
        v21 = (__int64 **)&v43[16 * v17 - 16];
        v22 = (__int64)v21[1];
        v23 = *v21;
        LODWORD(v44) = v17 - 1;
        v24 = *(_BYTE *)(v22 + 16);
        switch ( v24 )
        {
          case '6':
            v18 = *(_QWORD *)v22;
LABEL_20:
            v19 = (unsigned __int64)(sub_127FA20(v16, v18) + 7) >> 3;
            if ( v19 < *a2 )
              v19 = *a2;
            *a2 = v19;
            goto LABEL_23;
          case '7':
            v25 = *(__int64 **)(v22 - 48);
            if ( v25 == v23 )
              goto LABEL_46;
            v18 = *v25;
            goto LABEL_20;
          case '8':
            if ( !(unsigned __int8)sub_15FA1F0(v22) )
            {
              v20 = v43;
              goto LABEL_46;
            }
            break;
          default:
            if ( (unsigned __int8)(v24 - 71) > 1u )
              goto LABEL_46;
            break;
        }
        v26 = *(_QWORD *)(v22 + 8);
        if ( v26 )
        {
          while ( 1 )
          {
            v30 = sub_1648700(v26);
            v31 = v38;
            if ( v39 == v38 )
            {
              v32 = &v38[8 * HIDWORD(v40)];
              if ( v38 != (_BYTE *)v32 )
              {
                v33 = 0;
                while ( v30 != (_QWORD *)*v31 )
                {
                  if ( *v31 == -2 )
                    v33 = v31;
                  if ( v32 == ++v31 )
                  {
                    if ( !v33 )
                      goto LABEL_54;
                    *v33 = v30;
                    --v41;
                    ++v37;
                    goto LABEL_43;
                  }
                }
                goto LABEL_33;
              }
LABEL_54:
              if ( HIDWORD(v40) < (unsigned int)v40 )
                break;
            }
            sub_16CCBA0((__int64)&v37, (__int64)v30);
            if ( v29 )
            {
LABEL_43:
              v34 = (unsigned int)v44;
              if ( (unsigned int)v44 < HIDWORD(v44) )
              {
LABEL_44:
                v35 = (__int64 *)&v43[16 * v34];
                *v35 = v22;
                v35[1] = (__int64)v30;
                LODWORD(v44) = v44 + 1;
                goto LABEL_33;
              }
LABEL_56:
              sub_16CD150((__int64)&v43, v45, 0, 16, v27, v28);
              v34 = (unsigned int)v44;
              goto LABEL_44;
            }
LABEL_33:
            v26 = *(_QWORD *)(v26 + 8);
            if ( !v26 )
              goto LABEL_23;
          }
          ++HIDWORD(v40);
          *v32 = v30;
          v34 = (unsigned int)v44;
          ++v37;
          if ( (unsigned int)v44 < HIDWORD(v44) )
            goto LABEL_44;
          goto LABEL_56;
        }
LABEL_23:
        v17 = v44;
      }
      while ( (_DWORD)v44 );
    }
    v20 = v43;
    v22 = 0;
LABEL_46:
    if ( v20 != v45 )
      _libc_free((unsigned __int64)v20);
  }
  else
  {
    v22 = 0;
  }
  if ( v39 != v38 )
    _libc_free((unsigned __int64)v39);
  return v22;
}
