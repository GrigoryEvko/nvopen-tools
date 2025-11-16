// Function: sub_11F2320
// Address: 0x11f2320
//
unsigned __int64 __fastcall sub_11F2320(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r14
  unsigned __int64 v7; // rsi
  _QWORD *v8; // rax
  _QWORD *v9; // r15
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // r15
  __int64 v19; // rdx
  _QWORD *v20; // rbx
  _QWORD *v21; // r12
  __int64 v22; // rdi
  unsigned int v24; // eax
  int v25; // ecx
  char v26; // cl
  __int64 *v27; // [rsp+0h] [rbp-F0h]
  char v28; // [rsp+Fh] [rbp-E1h]
  _QWORD *v29; // [rsp+10h] [rbp-E0h]
  __int64 v30; // [rsp+20h] [rbp-D0h]
  __int64 v31; // [rsp+28h] [rbp-C8h]
  unsigned int v32; // [rsp+3Ch] [rbp-B4h] BYREF
  _BYTE *v33; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+48h] [rbp-A8h]
  _BYTE v35[160]; // [rsp+50h] [rbp-A0h] BYREF

  v27 = (__int64 *)sub_B43CA0(a2);
  if ( ((unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 23) || (unsigned __int8)sub_B49560(a2, 23))
    && !(unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 4)
    && !(unsigned __int8)sub_B49560(a2, 4) )
  {
    return 0;
  }
  v6 = *(_QWORD *)(a2 - 32);
  if ( v6 )
  {
    if ( *(_BYTE *)v6 )
    {
      v6 = 0;
    }
    else if ( *(_QWORD *)(v6 + 24) != *(_QWORD *)(a2 + 80) )
    {
      v6 = 0;
    }
  }
  v28 = sub_97F320(a2);
  v33 = v35;
  v34 = 0x200000000LL;
  sub_B56970(a2, (__int64)&v33);
  v30 = *(_QWORD *)(a3 + 112);
  v31 = *(_QWORD *)(a3 + 120);
  *(_QWORD *)(a3 + 112) = v33;
  *(_QWORD *)(a3 + 120) = (unsigned int)v34;
  v29 = sub_C52410();
  v7 = sub_C959E0();
  v8 = (_QWORD *)v29[2];
  v9 = v29 + 1;
  if ( v8 )
  {
    v10 = v29 + 1;
    do
    {
      while ( 1 )
      {
        v11 = v8[2];
        v12 = v8[3];
        if ( v7 <= v8[4] )
          break;
        v8 = (_QWORD *)v8[3];
        if ( !v12 )
          goto LABEL_12;
      }
      v10 = v8;
      v8 = (_QWORD *)v8[2];
    }
    while ( v11 );
LABEL_12:
    if ( v9 != v10 && v7 >= v10[4] )
      v9 = v10;
  }
  if ( v9 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_111;
  v13 = v9[7];
  if ( !v13 )
    goto LABEL_111;
  v7 = (unsigned int)dword_4F91388;
  v14 = v9 + 6;
  do
  {
    while ( 1 )
    {
      v15 = *(_QWORD *)(v13 + 16);
      v16 = *(_QWORD *)(v13 + 24);
      if ( *(_DWORD *)(v13 + 32) >= dword_4F91388 )
        break;
      v13 = *(_QWORD *)(v13 + 24);
      if ( !v16 )
        goto LABEL_21;
    }
    v14 = (_QWORD *)v13;
    v13 = *(_QWORD *)(v13 + 16);
  }
  while ( v15 );
LABEL_21:
  if ( v9 + 6 == v14 || dword_4F91388 < *((_DWORD *)v14 + 8) || *((int *)v14 + 9) <= 0 )
  {
LABEL_111:
    if ( (unsigned __int8)sub_920620(a2) && sub_B45190(a2) )
      *(_BYTE *)(a1 + 80) = 1;
  }
  else
  {
    *(_BYTE *)(a1 + 80) = qword_4F91408;
  }
  v17 = *(_QWORD *)(a2 - 32);
  if ( v17 && !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v17 + 33) & 0x20) != 0 )
  {
    if ( v28 )
    {
      v24 = *(_DWORD *)(v17 + 36);
      if ( v24 == 241 )
      {
        v7 = a2;
        v18 = sub_11E3820(a1, a2, a3);
        goto LABEL_28;
      }
      if ( v24 > 0xF1 )
      {
        switch ( v24 )
        {
          case 0x11Cu:
            v7 = a2;
            v18 = sub_11EF7B0(a1, a2, a3);
            goto LABEL_28;
          case 0x14Fu:
            v7 = a2;
            v18 = sub_11EE900(a1, a2, (unsigned int **)a3);
            goto LABEL_28;
          case 0xF3u:
            v7 = a2;
            v18 = sub_11E3930(a1, a2, (__int64 *)a3);
            goto LABEL_28;
        }
      }
      else if ( v24 > 0xDC )
      {
        if ( v24 == 238 )
        {
          v7 = a2;
          v18 = sub_11E3340(a1, a2, a3);
          goto LABEL_28;
        }
      }
      else
      {
        if ( v24 > 0xD9 )
        {
          v7 = a2;
          v18 = sub_11EBE80(a1, a2, a3);
          goto LABEL_28;
        }
        if ( v24 == 90 )
        {
          v7 = a2;
          v18 = sub_11EA2A0(a1, a2, a3);
          goto LABEL_28;
        }
      }
    }
    goto LABEL_48;
  }
  v7 = a2;
  v18 = sub_11EDEC0((__int64 **)a1, a2, (__int64 *)a3);
  if ( !v18 )
  {
    v7 = v6;
    if ( sub_981210(**(_QWORD **)(a1 + 24), v6, &v32) )
    {
      v7 = *(_QWORD *)(a1 + 24);
      if ( sub_11C99B0(v27, (__int64 *)v7, v32) && (v32 == 158 || v32 == 327 || v32 == 468 || v32 == 332 || v28) )
      {
        v7 = a2;
        v18 = sub_11F17D0(a1, a2, (unsigned int **)a3);
        if ( v18 )
          goto LABEL_28;
        v7 = a2;
        v18 = (unsigned __int64)sub_11F1F40(a1, a2, v32, (__int64 *)a3);
        if ( v18 )
          goto LABEL_28;
        if ( v32 <= 0x14C )
        {
          if ( v32 > 0xF6 )
          {
            switch ( v32 )
            {
              case 0xF7u:
              case 0xF8u:
              case 0xF9u:
                v7 = a2;
                v18 = sub_11E57D0(a1, a2, a3);
                break;
              case 0x100u:
                goto LABEL_87;
              case 0x105u:
              case 0x106u:
              case 0x107u:
                v7 = a2;
                v18 = sub_11E5AD0(a1, a2, a3);
                break;
              case 0x11Au:
                v7 = a2;
                v18 = sub_11E9290(a1, a2, a3);
                break;
              case 0x11Du:
                v7 = a2;
                v18 = sub_11E97B0((_QWORD *)a1, a2, a3);
                break;
              case 0x133u:
                v7 = a2;
                v18 = sub_11E9510(a1, a2, a3);
                break;
              case 0x145u:
                v7 = a2;
                v18 = sub_11E6190(a1, a2, (__int64 *)a3);
                break;
              case 0x146u:
                v7 = a2;
                v18 = sub_11E5E20(a1, a2, a3);
                break;
              case 0x147u:
              case 0x14Cu:
                goto LABEL_101;
              default:
                goto LABEL_48;
            }
            goto LABEL_28;
          }
          if ( v32 == 158 )
          {
LABEL_101:
            v7 = a2;
            v18 = sub_11E5BF0(a1, a2, a3);
            goto LABEL_28;
          }
          if ( v32 <= 0x9E )
          {
            if ( v32 - 91 <= 1 )
            {
              v7 = a2;
              v18 = sub_11E99D0(a1, a2);
              goto LABEL_28;
            }
          }
          else if ( v32 - 183 <= 2 )
          {
            v7 = a2;
            v18 = sub_11E6620(a1, a2, a3);
            goto LABEL_28;
          }
          goto LABEL_48;
        }
        if ( v32 > 0x1F3 )
        {
          if ( v32 == 514 )
          {
LABEL_87:
            v25 = 0;
LABEL_88:
            v7 = a2;
            v18 = sub_11E6850(a1, (unsigned __int8 *)a2, a3, v25);
            goto LABEL_28;
          }
        }
        else
        {
          if ( v32 > 0x1BD )
          {
            switch ( v32 )
            {
              case 0x1BEu:
                v7 = a2;
                v18 = sub_11E8D30(a1, a2, a3);
                goto LABEL_28;
              case 0x1BFu:
                v7 = a2;
                v18 = sub_11E7FA0(a1, a2, a3);
                goto LABEL_28;
              case 0x1E3u:
              case 0x1E5u:
                v26 = 1;
                goto LABEL_97;
              case 0x1E6u:
              case 0x1E7u:
                v26 = 0;
LABEL_97:
                v7 = a2;
                v18 = sub_11E66A0(a1, a2, a3, v26);
                break;
              case 0x1F3u:
                v7 = a2;
                v18 = sub_11E64B0(a1, a2, (__int64 *)a3);
                break;
              default:
                goto LABEL_48;
            }
            goto LABEL_28;
          }
          switch ( v32 )
          {
            case 0x186u:
              v7 = a2;
              v18 = sub_11E7230(a1, a2, a3);
              goto LABEL_28;
            case 0x18Bu:
              v7 = a2;
              v18 = sub_11E98F0(a1, a2, a3);
              goto LABEL_28;
            case 0x17Fu:
              v25 = -1;
              goto LABEL_88;
          }
        }
      }
    }
LABEL_48:
    v18 = 0;
  }
LABEL_28:
  v19 = (unsigned int)v34;
  *(_QWORD *)(a3 + 112) = v30;
  *(_QWORD *)(a3 + 120) = v31;
  v20 = v33;
  v21 = &v33[56 * v19];
  if ( v33 != (_BYTE *)v21 )
  {
    do
    {
      v22 = *(v21 - 3);
      v21 -= 7;
      if ( v22 )
      {
        v7 = v21[6] - v22;
        j_j___libc_free_0(v22, v7);
      }
      if ( (_QWORD *)*v21 != v21 + 2 )
      {
        v7 = v21[2] + 1LL;
        j_j___libc_free_0(*v21, v7);
      }
    }
    while ( v20 != v21 );
    v21 = v33;
  }
  if ( v21 != (_QWORD *)v35 )
    _libc_free(v21, v7);
  return v18;
}
