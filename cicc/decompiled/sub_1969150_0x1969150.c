// Function: sub_1969150
// Address: 0x1969150
//
__int64 __fastcall sub_1969150(
        __int64 a1,
        unsigned __int8 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        _QWORD *a6,
        __int64 a7)
{
  __int64 v7; // rax
  _QWORD *v8; // r12
  __int64 v9; // rcx
  __int64 v10; // r13
  __int64 v11; // rbx
  _QWORD *v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  _QWORD *v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdi
  char v19; // al
  _QWORD *v21; // rdx
  __int64 v22; // [rsp+0h] [rbp-90h]
  __int64 v24; // [rsp+10h] [rbp-80h]
  __int64 v27; // [rsp+28h] [rbp-68h]
  _QWORD v28[12]; // [rsp+30h] [rbp-60h] BYREF

  v27 = -1;
  if ( !*(_WORD *)(a4 + 24) )
  {
    v7 = *(_QWORD *)(a4 + 32);
    v8 = *(_QWORD **)(v7 + 24);
    if ( *(_DWORD *)(v7 + 32) > 0x40u )
      v8 = (_QWORD *)*v8;
    v27 = a5 * ((_QWORD)v8 + 1);
  }
  v9 = *(_QWORD *)(a3 + 40);
  v24 = *(_QWORD *)(a3 + 32);
  v22 = v9;
  if ( v9 != v24 )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(*(_QWORD *)v24 + 48LL);
      v11 = *(_QWORD *)v24 + 40LL;
      if ( v11 != v10 )
        break;
LABEL_33:
      v24 += 8;
      if ( v24 == v22 )
        return 0;
    }
    while ( 1 )
    {
      v15 = 0;
      v16 = *(_QWORD **)(a7 + 16);
      if ( v10 )
        v15 = v10 - 24;
      v13 = *(_QWORD **)(a7 + 8);
      if ( v16 == v13 )
      {
        v12 = &v13[*(unsigned int *)(a7 + 28)];
        if ( v13 == v12 )
        {
          v21 = *(_QWORD **)(a7 + 8);
        }
        else
        {
          do
          {
            if ( v15 == *v13 )
              break;
            ++v13;
          }
          while ( v12 != v13 );
          v21 = v12;
        }
      }
      else
      {
        v12 = &v16[*(unsigned int *)(a7 + 24)];
        v13 = sub_16CC9F0(a7, v15);
        if ( v15 == *v13 )
        {
          v17 = *(_QWORD *)(a7 + 16);
          if ( v17 == *(_QWORD *)(a7 + 8) )
            v18 = *(unsigned int *)(a7 + 28);
          else
            v18 = *(unsigned int *)(a7 + 24);
          v21 = (_QWORD *)(v17 + 8 * v18);
        }
        else
        {
          v14 = *(_QWORD *)(a7 + 16);
          if ( v14 != *(_QWORD *)(a7 + 8) )
          {
            v13 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a7 + 24));
            goto LABEL_11;
          }
          v13 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a7 + 28));
          v21 = v13;
        }
      }
      if ( v13 != v21 )
      {
        while ( *v13 >= 0xFFFFFFFFFFFFFFFELL )
        {
          if ( v21 == ++v13 )
          {
            if ( v12 != v13 )
              goto LABEL_12;
            goto LABEL_25;
          }
        }
      }
LABEL_11:
      if ( v12 == v13 )
      {
LABEL_25:
        memset(&v28[2], 0, 24);
        v28[0] = a1;
        v28[1] = v27;
        switch ( *(_BYTE *)(v15 + 16) )
        {
          case 0x1D:
            v19 = sub_134F0E0(a6, v15 & 0xFFFFFFFFFFFFFFFBLL, (__int64)v28);
            break;
          case 0x21:
            v19 = sub_134D290((__int64)a6, v15, v28);
            break;
          case 0x36:
            v19 = sub_134D040((__int64)a6, v15, v28, v9);
            break;
          case 0x37:
            v19 = sub_134D0E0((__int64)a6, v15, v28, v9);
            break;
          case 0x39:
            v19 = sub_134D190((__int64)a6, v15, v28);
            break;
          case 0x3A:
            v19 = sub_134D2D0((__int64)a6, v15, v28);
            break;
          case 0x3B:
            v19 = sub_134D360((__int64)a6, v15, v28);
            break;
          case 0x4A:
            v19 = sub_134D250((__int64)a6, v15, v28);
            break;
          case 0x4E:
            v19 = sub_134F0E0(a6, v15 | 4, (__int64)v28);
            break;
          case 0x52:
            v19 = sub_134D1D0((__int64)a6, v15, v28);
            break;
          default:
            v19 = 4;
            break;
        }
        if ( (a2 & (unsigned __int8)v19 & 3) != 0 )
          return 1;
      }
LABEL_12:
      v10 = *(_QWORD *)(v10 + 8);
      if ( v11 == v10 )
        goto LABEL_33;
    }
  }
  return 0;
}
