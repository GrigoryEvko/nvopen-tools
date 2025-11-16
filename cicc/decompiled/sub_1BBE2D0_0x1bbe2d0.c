// Function: sub_1BBE2D0
// Address: 0x1bbe2d0
//
__int64 __fastcall sub_1BBE2D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int8 v6; // al
  __int64 **v8; // rsi
  _QWORD *v9; // rcx
  __int64 **v10; // rdi
  unsigned int v11; // eax
  __int64 *v12; // r12
  int v13; // eax
  unsigned __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 *v17; // r14
  __int64 **v18; // r15
  __int64 **v19; // rax
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 **v22; // rcx
  __int64 **v23; // rdx
  unsigned int v24; // eax
  __int64 **v25; // rdx
  unsigned int v28; // [rsp+2Ch] [rbp-174h]
  _QWORD *v29; // [rsp+30h] [rbp-170h] BYREF
  __int64 v30; // [rsp+38h] [rbp-168h]
  _QWORD v31[16]; // [rsp+40h] [rbp-160h] BYREF
  __int64 v32; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 **v33; // [rsp+C8h] [rbp-D8h]
  __int64 **v34; // [rsp+D0h] [rbp-D0h]
  __int64 v35; // [rsp+D8h] [rbp-C8h]
  int v36; // [rsp+E0h] [rbp-C0h]
  _BYTE v37[184]; // [rsp+E8h] [rbp-B8h] BYREF

  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 > 0x17u )
  {
    if ( v6 == 55 )
      return (unsigned int)sub_127FA20(*(_QWORD *)(a1 + 1376), **(_QWORD **)(a2 - 48));
    v8 = (__int64 **)v37;
    v32 = 0;
    v33 = (__int64 **)v37;
    v34 = (__int64 **)v37;
    v9 = v31;
    v35 = 16;
    v36 = 0;
    v28 = 0;
    v31[0] = a2;
    v29 = v31;
    v30 = 0x1000000001LL;
    v10 = (__int64 **)v37;
    v11 = 1;
    while ( 1 )
    {
      v12 = (__int64 *)v9[v11 - 1];
      LODWORD(v30) = v11 - 1;
      if ( v10 != v8 )
        break;
      v22 = &v8[HIDWORD(v35)];
      if ( v22 == v8 )
      {
LABEL_60:
        if ( HIDWORD(v35) >= (unsigned int)v35 )
          break;
        ++HIDWORD(v35);
        *v22 = v12;
        ++v32;
      }
      else
      {
        v23 = 0;
        while ( v12 != *v8 )
        {
          if ( *v8 == (__int64 *)-2LL )
            v23 = v8;
          if ( v22 == ++v8 )
          {
            if ( !v23 )
              goto LABEL_60;
            *v23 = v12;
            --v36;
            ++v32;
            break;
          }
        }
      }
LABEL_13:
      if ( *(_BYTE *)(*v12 + 8) == 16 )
        goto LABEL_3;
      v13 = *((unsigned __int8 *)v12 + 16);
      if ( (_BYTE)v13 != 54 )
      {
        if ( (_BYTE)v13 != 77 )
        {
          v14 = (unsigned int)(v13 - 35);
          if ( (unsigned __int8)v14 > 0x2Cu )
            goto LABEL_3;
          v15 = 0x133FFE23FFFFLL;
          if ( !_bittest64(&v15, v14) )
            goto LABEL_3;
        }
        v16 = 24LL * (*((_DWORD *)v12 + 5) & 0xFFFFFFF);
        if ( (*((_BYTE *)v12 + 23) & 0x40) != 0 )
        {
          v17 = (__int64 *)*(v12 - 1);
          v12 = &v17[(unsigned __int64)v16 / 8];
        }
        else
        {
          v17 = &v12[v16 / 0xFFFFFFFFFFFFFFF8LL];
        }
        if ( v17 == v12 )
        {
LABEL_41:
          v11 = v30;
          if ( !(_DWORD)v30 )
            goto LABEL_58;
          goto LABEL_42;
        }
        while ( 1 )
        {
          v20 = *v17;
          if ( *(_BYTE *)(*v17 + 16) <= 0x17u )
          {
LABEL_26:
            v17 += 3;
            if ( v12 == v17 )
              goto LABEL_41;
            continue;
          }
          v19 = v33;
          if ( v34 == v33 )
          {
            v18 = &v33[HIDWORD(v35)];
            if ( v33 == v18 )
            {
              v25 = v33;
            }
            else
            {
              do
              {
                if ( (__int64 *)v20 == *v19 )
                  break;
                ++v19;
              }
              while ( v18 != v19 );
              v25 = &v33[HIDWORD(v35)];
            }
          }
          else
          {
            v18 = &v34[(unsigned int)v35];
            v19 = (__int64 **)sub_16CC9F0((__int64)&v32, *v17);
            if ( (__int64 *)v20 == *v19 )
            {
              if ( v34 == v33 )
                v25 = &v34[HIDWORD(v35)];
              else
                v25 = &v34[(unsigned int)v35];
            }
            else
            {
              if ( v34 != v33 )
              {
                v19 = &v34[(unsigned int)v35];
                goto LABEL_25;
              }
              v19 = &v34[HIDWORD(v35)];
              v25 = v19;
            }
          }
          if ( v19 != v25 )
          {
            while ( (unsigned __int64)*v19 >= 0xFFFFFFFFFFFFFFFELL )
            {
              if ( v25 == ++v19 )
              {
                if ( v18 != v19 )
                  goto LABEL_26;
                goto LABEL_38;
              }
            }
          }
LABEL_25:
          if ( v18 != v19 )
            goto LABEL_26;
LABEL_38:
          v21 = (unsigned int)v30;
          if ( (unsigned int)v30 >= HIDWORD(v30) )
          {
            sub_16CD150((__int64)&v29, v31, 0, 8, a5, a6);
            v21 = (unsigned int)v30;
          }
          v17 += 3;
          v29[v21] = v20;
          LODWORD(v30) = v30 + 1;
          if ( v12 == v17 )
            goto LABEL_41;
        }
      }
      v24 = sub_127FA20(*(_QWORD *)(a1 + 1376), *v12);
      if ( v28 >= v24 )
        v24 = v28;
      v28 = v24;
      v11 = v30;
      if ( !(_DWORD)v30 )
      {
LABEL_58:
        if ( v28 )
          goto LABEL_4;
        goto LABEL_3;
      }
LABEL_42:
      v9 = v29;
      v10 = v34;
      v8 = v33;
    }
    sub_16CCBA0((__int64)&v32, (__int64)v12);
    goto LABEL_13;
  }
  v32 = 0;
  v29 = v31;
  v30 = 0x1000000000LL;
  v33 = (__int64 **)v37;
  v34 = (__int64 **)v37;
  v35 = 16;
  v36 = 0;
LABEL_3:
  v28 = sub_127FA20(*(_QWORD *)(a1 + 1376), *(_QWORD *)a2);
LABEL_4:
  if ( v34 != v33 )
    _libc_free((unsigned __int64)v34);
  if ( v29 != v31 )
    _libc_free((unsigned __int64)v29);
  return v28;
}
