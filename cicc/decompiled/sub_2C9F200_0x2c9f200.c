// Function: sub_2C9F200
// Address: 0x2c9f200
//
__int64 __fastcall sub_2C9F200(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 result; // rax
  _QWORD *v6; // r15
  _QWORD *v7; // r12
  __int64 v8; // rcx
  int v9; // r13d
  __int64 v10; // r15
  unsigned __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // r14
  _BYTE *v16; // rsi
  __int64 v17; // rdx
  _QWORD *v18; // r13
  __int64 v19; // rcx
  int v20; // r12d
  __int64 v21; // r15
  unsigned __int64 v22; // rax
  int v23; // edx
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // r14
  _BYTE *v27; // rsi
  __int64 v28; // [rsp+8h] [rbp-88h]
  int v29; // [rsp+14h] [rbp-7Ch]
  __int64 v31; // [rsp+20h] [rbp-70h]
  __int64 v32; // [rsp+28h] [rbp-68h]
  __int64 v33; // [rsp+28h] [rbp-68h]
  __int64 v34; // [rsp+48h] [rbp-48h] BYREF
  __int64 v35; // [rsp+50h] [rbp-40h] BYREF
  __int64 v36[7]; // [rsp+58h] [rbp-38h] BYREF

  v3 = *a2;
  result = (__int64)(*(_QWORD *)(*(_QWORD *)*a2 + 8LL) - **(_QWORD **)*a2) >> 3;
  if ( !(_DWORD)result )
    return result;
  v29 = result - 1;
  v28 = (unsigned int)result;
  v31 = 0;
  v6 = a2;
  while ( 2 )
  {
    v34 = 0;
    if ( v6[1] == v3 )
      goto LABEL_18;
    v7 = v6;
    v8 = 0;
    v9 = 0;
    v10 = a1;
    do
    {
      v15 = *(_QWORD *)(**(_QWORD **)(v3 + 8 * v8) + 8 * v31);
      sub_2C9EEF0(v10, *(unsigned int **)v15, *(_QWORD ***)(v15 + 8), &v35, v36, *(_QWORD *)(v10 + 200), 0);
      v13 = v35;
      if ( *(_BYTE *)v35 == 23 )
      {
        if ( v35 == *(_QWORD *)(*(_QWORD *)(***(_QWORD ***)v15 + 16LL) + 40LL) )
        {
          v13 = *(_QWORD *)(***(_QWORD ***)v15 + 16LL);
        }
        else
        {
          v11 = *(_QWORD *)(v35 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v11 == v35 + 48 )
          {
            v13 = 0;
          }
          else
          {
            if ( !v11 )
              BUG();
            v12 = *(unsigned __int8 *)(v11 - 24);
            v13 = 0;
            v14 = v11 - 24;
            if ( (unsigned int)(v12 - 30) < 0xB )
              v13 = v14;
          }
        }
      }
      if ( v34 )
      {
        v32 = v13;
        if ( !(unsigned __int8)sub_B19DB0(*(_QWORD *)(v10 + 200), v13, v34) )
          goto LABEL_13;
        v13 = v32;
      }
      v34 = v13;
LABEL_13:
      v3 = *v7;
      v8 = (unsigned int)++v9;
    }
    while ( v9 != (__int64)(v7[1] - *v7) >> 3 );
    a1 = v10;
    v6 = v7;
LABEL_18:
    v16 = *(_BYTE **)(a3 + 8);
    if ( v16 == *(_BYTE **)(a3 + 16) )
    {
      sub_24454E0(a3, v16, &v34);
    }
    else
    {
      if ( v16 )
      {
        *(_QWORD *)v16 = v34;
        v16 = *(_BYTE **)(a3 + 8);
      }
      *(_QWORD *)(a3 + 8) = v16 + 8;
    }
    if ( v29 != (_DWORD)v31 )
      goto LABEL_23;
    v34 = 0;
    v17 = *v6;
    if ( *v6 != v6[1] )
    {
      v18 = v6;
      v19 = 0;
      v20 = 0;
      v21 = a1;
      while ( 1 )
      {
        v26 = *(_QWORD *)(**(_QWORD **)(v17 + 8 * v19) + 8 * v31);
        sub_2C9EEF0(v21, *(unsigned int **)v26, *(_QWORD ***)(v26 + 8), &v35, v36, *(_QWORD *)(v21 + 200), 0);
        v24 = v36[0];
        if ( *(_BYTE *)v36[0] == 23 )
        {
          if ( v36[0] == *(_QWORD *)(*(_QWORD *)(***(_QWORD ***)(v26 + 8) + 16LL) + 40LL) )
          {
            v24 = *(_QWORD *)(***(_QWORD ***)(v26 + 8) + 16LL);
          }
          else
          {
            v22 = *(_QWORD *)(v36[0] + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v22 == v36[0] + 48 )
            {
              v24 = 0;
            }
            else
            {
              if ( !v22 )
                BUG();
              v23 = *(unsigned __int8 *)(v22 - 24);
              v24 = 0;
              v25 = v22 - 24;
              if ( (unsigned int)(v23 - 30) < 0xB )
                v24 = v25;
            }
          }
        }
        if ( !v34 )
          goto LABEL_34;
        v33 = v24;
        if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(v21 + 200), v24, v34) )
          break;
LABEL_35:
        v17 = *v18;
        v19 = (unsigned int)++v20;
        if ( v20 == (__int64)(v18[1] - *v18) >> 3 )
        {
          a1 = v21;
          v6 = v18;
          goto LABEL_40;
        }
      }
      v24 = v33;
LABEL_34:
      v34 = v24;
      goto LABEL_35;
    }
LABEL_40:
    v27 = *(_BYTE **)(a3 + 8);
    if ( v27 == *(_BYTE **)(a3 + 16) )
    {
      sub_24454E0(a3, v27, &v34);
    }
    else
    {
      if ( v27 )
      {
        *(_QWORD *)v27 = v34;
        v27 = *(_BYTE **)(a3 + 8);
      }
      *(_QWORD *)(a3 + 8) = v27 + 8;
    }
LABEL_23:
    result = ++v31;
    if ( v28 != v31 )
    {
      v3 = *v6;
      continue;
    }
    return result;
  }
}
