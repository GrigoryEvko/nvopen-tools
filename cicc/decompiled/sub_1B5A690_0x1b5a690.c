// Function: sub_1B5A690
// Address: 0x1b5a690
//
__int64 __fastcall sub_1B5A690(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // r12
  unsigned __int64 v5; // rax
  int v6; // ebx
  unsigned __int64 v7; // r13
  unsigned int v8; // r12d
  __int64 v9; // rsi
  __int64 *v10; // rax
  __int64 *v11; // rdi
  __int64 *v12; // rcx
  unsigned __int64 v13; // rdi
  __int64 v14; // r15
  unsigned int v15; // ebx
  __int64 v16; // r12
  __int64 v17; // r8
  __int64 *v18; // r9
  __int64 *v19; // rax
  __int64 *v20; // r14
  __int64 v21; // rdx
  __int64 v22; // r14
  __int64 v23; // r15
  __int64 i; // r12
  __int64 v25; // r11
  __int64 v26; // rax
  char v27; // di
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned int v35; // r12d
  int v38; // [rsp+14h] [rbp-10Ch]
  unsigned __int64 v39; // [rsp+18h] [rbp-108h]
  __int64 v40; // [rsp+20h] [rbp-100h]
  __int64 v41; // [rsp+28h] [rbp-F8h]
  unsigned __int8 v42; // [rsp+28h] [rbp-F8h]
  __int64 v43; // [rsp+38h] [rbp-E8h] BYREF
  __int64 v44; // [rsp+40h] [rbp-E0h] BYREF
  __int64 *v45; // [rsp+48h] [rbp-D8h]
  __int64 *v46; // [rsp+50h] [rbp-D0h]
  __int64 v47; // [rsp+58h] [rbp-C8h]
  int v48; // [rsp+60h] [rbp-C0h]
  _BYTE v49[184]; // [rsp+68h] [rbp-B8h] BYREF

  v3 = *(_QWORD *)(a1 + 40);
  v4 = *(_QWORD *)(a2 + 40);
  v5 = sub_157EBA0(v3);
  if ( !v5 )
  {
    v44 = 0;
    v45 = (__int64 *)v49;
    v46 = (__int64 *)v49;
    v47 = 16;
    v48 = 0;
    goto LABEL_16;
  }
  v6 = sub_15F4D60(v5);
  v44 = 0;
  v47 = 16;
  v7 = sub_157EBA0(v3);
  v45 = (__int64 *)v49;
  v46 = (__int64 *)v49;
  v48 = 0;
  if ( v6 )
  {
    v41 = v4;
    v8 = 0;
    while ( 1 )
    {
LABEL_6:
      v9 = sub_15F4DF0(v7, v8);
      v10 = v45;
      if ( v46 != v45 )
        goto LABEL_4;
      v11 = &v45[HIDWORD(v47)];
      if ( v45 != v11 )
      {
        v12 = 0;
        while ( v9 != *v10 )
        {
          if ( *v10 == -2 )
            v12 = v10;
          if ( v11 == ++v10 )
          {
            if ( !v12 )
              goto LABEL_66;
            ++v8;
            *v12 = v9;
            --v48;
            ++v44;
            if ( v6 != v8 )
              goto LABEL_6;
            goto LABEL_15;
          }
        }
        goto LABEL_5;
      }
LABEL_66:
      if ( HIDWORD(v47) < (unsigned int)v47 )
      {
        ++HIDWORD(v47);
        *v11 = v9;
        ++v44;
      }
      else
      {
LABEL_4:
        sub_16CCBA0((__int64)&v44, v9);
      }
LABEL_5:
      if ( v6 == ++v8 )
      {
LABEL_15:
        v4 = v41;
        break;
      }
    }
  }
LABEL_16:
  v13 = sub_157EBA0(v4);
  if ( v13 )
  {
    v38 = sub_15F4D60(v13);
    v39 = sub_157EBA0(v4);
    if ( v38 )
    {
      v42 = 0;
      v14 = v4;
      v15 = 0;
      v16 = v3;
      while ( 1 )
      {
        v43 = sub_15F4DF0(v39, v15);
        v19 = v45;
        if ( v46 == v45 )
        {
          v20 = &v45[HIDWORD(v47)];
          if ( v45 == v20 )
          {
            v21 = (__int64)v45;
          }
          else
          {
            do
            {
              if ( v43 == *v19 )
                break;
              ++v19;
            }
            while ( v20 != v19 );
            v21 = (__int64)&v45[HIDWORD(v47)];
          }
          goto LABEL_61;
        }
        v40 = v43;
        v20 = &v46[(unsigned int)v47];
        v19 = sub_16CC9F0((__int64)&v44, v43);
        if ( v40 == *v19 )
          break;
        if ( v46 == v45 )
        {
          v19 = &v46[HIDWORD(v47)];
          v21 = (__int64)v19;
          goto LABEL_61;
        }
        v21 = (unsigned int)v47;
        v19 = &v46[(unsigned int)v47];
LABEL_23:
        if ( v20 != v19 )
        {
          v22 = v14;
          v23 = v16;
          for ( i = *(_QWORD *)(v43 + 48); ; i = *(_QWORD *)(i + 8) )
          {
            if ( !i )
              BUG();
            if ( *(_BYTE *)(i - 8) != 77 )
              break;
            v25 = i - 24;
            v26 = 0x17FFFFFFE8LL;
            v27 = *(_BYTE *)(i - 1) & 0x40;
            v28 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
            if ( (*(_DWORD *)(i - 4) & 0xFFFFFFF) != 0 )
            {
              v21 = 24LL * *(unsigned int *)(i + 32) + 8;
              v29 = 0;
              do
              {
                v30 = v25 - 24LL * (unsigned int)v28;
                if ( v27 )
                  v30 = *(_QWORD *)(i - 32);
                if ( v23 == *(_QWORD *)(v30 + v21) )
                {
                  v26 = 24 * v29;
                  goto LABEL_34;
                }
                ++v29;
                v21 += 8;
              }
              while ( (_DWORD)v28 != (_DWORD)v29 );
              v26 = 0x17FFFFFFE8LL;
            }
LABEL_34:
            if ( v27 )
            {
              v31 = *(_QWORD *)(i - 32);
            }
            else
            {
              v21 = 24LL * (unsigned int)v28;
              v31 = v25 - v21;
            }
            v32 = *(_QWORD *)(v31 + v26);
            v33 = 0x17FFFFFFE8LL;
            if ( (_DWORD)v28 )
            {
              v34 = 0;
              v21 = v31 + 24LL * *(unsigned int *)(i + 32);
              do
              {
                if ( v22 == *(_QWORD *)(v21 + 8 * v34 + 8) )
                {
                  v33 = 24 * v34;
                  goto LABEL_41;
                }
                ++v34;
              }
              while ( (_DWORD)v28 != (_DWORD)v34 );
              v33 = 0x17FFFFFFE8LL;
            }
LABEL_41:
            if ( v32 != *(_QWORD *)(v31 + v33) )
            {
              v42 = 1;
              if ( a3 )
                sub_1B5A4D0(a3, &v43, v21, v28, v17, v18);
            }
          }
          v16 = v23;
          v14 = v22;
        }
        if ( ++v15 == v38 )
        {
          v35 = v42 ^ 1;
          goto LABEL_51;
        }
      }
      if ( v46 == v45 )
        v21 = (__int64)&v46[HIDWORD(v47)];
      else
        v21 = (__int64)&v46[(unsigned int)v47];
LABEL_61:
      while ( (__int64 *)v21 != v19 && (unsigned __int64)*v19 >= 0xFFFFFFFFFFFFFFFELL )
        ++v19;
      goto LABEL_23;
    }
  }
  v35 = 1;
LABEL_51:
  if ( v46 != v45 )
    _libc_free((unsigned __int64)v46);
  return v35;
}
