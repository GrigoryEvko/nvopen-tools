// Function: sub_1563330
// Address: 0x1563330
//
__int64 __fastcall sub_1563330(__int64 *a1, __int64 *a2, int a3, _QWORD *a4)
{
  __int64 v4; // r12
  unsigned int v6; // ebx
  __int64 v7; // r15
  __int64 *v8; // rdi
  const void *v9; // r11
  int v10; // edx
  signed __int64 v11; // r8
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v16; // r8
  unsigned int v17; // ebx
  __int64 *v18; // rax
  __int64 *v19; // rcx
  signed __int64 v20; // [rsp+8h] [rbp-88h]
  const void *v21; // [rsp+10h] [rbp-80h]
  __int64 v22; // [rsp+20h] [rbp-70h]
  __int64 *v24; // [rsp+30h] [rbp-60h] BYREF
  __int64 v25; // [rsp+38h] [rbp-58h]
  _BYTE dest[80]; // [rsp+40h] [rbp-50h] BYREF

  v4 = 0;
  if ( !*a1 )
    return v4;
  if ( a3 == -1 )
  {
    v22 = 0;
    v6 = 0;
  }
  else
  {
    v6 = a3 + 1;
    v22 = (unsigned int)(a3 + 1);
  }
  v7 = sub_15601B0(a1);
  v24 = (__int64 *)dest;
  v8 = (__int64 *)dest;
  v9 = (const void *)sub_15601A0(a1);
  v10 = 0;
  v11 = v7 - (_QWORD)v9;
  v25 = 0x400000000LL;
  v12 = (v7 - (__int64)v9) >> 3;
  if ( (unsigned __int64)(v7 - (_QWORD)v9) > 0x20 )
  {
    v20 = v7 - (_QWORD)v9;
    v21 = v9;
    sub_16CD150(&v24, dest, v11 >> 3, 8);
    v10 = v25;
    v11 = v20;
    v9 = v21;
    v8 = &v24[(unsigned int)v25];
  }
  if ( (const void *)v7 != v9 )
  {
    memcpy(v8, v9, v11);
    v10 = v25;
  }
  LODWORD(v13) = v12 + v10;
  LODWORD(v25) = v12 + v10;
  if ( v6 >= (int)v12 + v10 )
  {
    v13 = (unsigned int)v13;
    v17 = v6 + 1;
    v16 = v17;
    if ( v17 < (unsigned __int64)(unsigned int)v13 )
    {
      LODWORD(v25) = v17;
    }
    else if ( v17 > (unsigned __int64)(unsigned int)v13 )
    {
      if ( v17 > (unsigned __int64)HIDWORD(v25) )
      {
        sub_16CD150(&v24, dest, v17, 8);
        v13 = (unsigned int)v25;
        v16 = v17;
      }
      v14 = v24;
      v18 = &v24[v13];
      v19 = &v24[v16];
      if ( v18 != v19 )
      {
        do
        {
          if ( v18 )
            *v18 = 0;
          ++v18;
        }
        while ( v19 != v18 );
        v14 = v24;
      }
      LODWORD(v25) = v17;
      goto LABEL_10;
    }
  }
  v14 = v24;
LABEL_10:
  v14[v22] = sub_15632D0(&v14[v22], a2, a4);
  v4 = sub_155F990(a2, v24, (unsigned int)v25);
  if ( v24 != (__int64 *)dest )
    _libc_free((unsigned __int64)v24);
  return v4;
}
