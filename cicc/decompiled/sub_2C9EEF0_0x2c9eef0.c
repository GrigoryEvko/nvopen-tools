// Function: sub_2C9EEF0
// Address: 0x2c9eef0
//
__int64 __fastcall sub_2C9EEF0(
        __int64 a1,
        unsigned int *a2,
        _QWORD **a3,
        __int64 *a4,
        __int64 *a5,
        __int64 a6,
        _BYTE *a7)
{
  __int64 *v8; // rax
  _QWORD *v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rbx
  __int64 v13; // r13
  unsigned int v14; // eax
  unsigned int v15; // edx
  int v16; // r15d
  __int64 v17; // rsi
  unsigned int v18; // eax
  unsigned int v19; // edx
  __int64 v20; // rcx
  __int64 v21; // rsi
  unsigned int v22; // eax
  __int64 *v23; // rax
  char v24; // al
  _BYTE *v25; // rsi
  unsigned __int64 v28; // rsi
  __int64 *v29; // rax
  _QWORD *v30; // rax
  _QWORD *v31; // r10
  unsigned __int64 v32; // r8
  unsigned __int64 v33; // rcx
  __int64 v34; // r11
  _QWORD *v35; // rdx
  _QWORD *v36; // rdi
  _BYTE *v37; // rsi
  unsigned __int64 v38; // [rsp+0h] [rbp-70h]
  __int64 v39; // [rsp+10h] [rbp-60h]
  unsigned __int64 v43[7]; // [rsp+38h] [rbp-38h] BYREF

  v8 = *(__int64 **)a2;
  v9 = (_QWORD *)**a3;
  v10 = *(_QWORD *)(**(_QWORD **)a2 + 16LL);
  v11 = v9[2];
  v39 = **(_QWORD **)a2;
  v12 = *(_QWORD *)(v10 + 40);
  v13 = *(_QWORD *)(v11 + 40);
  if ( v12 != v13 )
  {
    sub_B196A0(a6, v12, *(_QWORD *)(v11 + 40));
    v15 = v14;
    if ( !(_BYTE)v14 )
    {
      if ( (int)qword_5012D68 > 4 )
        goto LABEL_4;
      v38 = v9[1];
      v43[0] = *(_QWORD *)(v39 + 8);
      v29 = sub_2C97130(a1 + 32, (__int64 *)v43);
      v15 = 0;
      if ( *(_DWORD *)v29 <= 0x14u )
        return v15;
      v30 = *(_QWORD **)(a1 + 80);
      v31 = (_QWORD *)(a1 + 72);
      v32 = v43[0];
      v33 = v38;
      v34 = a1 + 72;
      if ( v30 )
      {
        v35 = *(_QWORD **)(a1 + 80);
        do
        {
          if ( v35[4] < v43[0] )
          {
            v35 = (_QWORD *)v35[3];
          }
          else
          {
            v31 = v35;
            v35 = (_QWORD *)v35[2];
          }
        }
        while ( v35 );
        if ( v31 != (_QWORD *)v34 && v31[4] <= v43[0] )
          v32 = v31[5];
        v36 = (_QWORD *)(a1 + 72);
        do
        {
          if ( v30[4] < v38 )
          {
            v30 = (_QWORD *)v30[3];
          }
          else
          {
            v36 = v30;
            v30 = (_QWORD *)v30[2];
          }
        }
        while ( v30 );
        if ( (_QWORD *)v34 != v36 && v36[4] <= v38 )
          v33 = v36[5];
      }
      if ( v33 == v32 )
      {
LABEL_4:
        v16 = 0;
        while ( ++v16 <= (unsigned int)qword_5011DA8 )
        {
          if ( v12 )
          {
            v17 = (unsigned int)(*(_DWORD *)(v12 + 44) + 1);
            v18 = *(_DWORD *)(v12 + 44) + 1;
          }
          else
          {
            v17 = 0;
            v18 = 0;
          }
          v19 = *(_DWORD *)(a6 + 32);
          if ( v18 >= v19 )
            BUG();
          v20 = *(_QWORD *)(a6 + 24);
          v12 = *(_QWORD *)(*(_QWORD *)(v20 + 8 * v17) + 8LL);
          if ( v12 )
            v12 = *(_QWORD *)v12;
          if ( v13 )
          {
            v21 = (unsigned int)(*(_DWORD *)(v13 + 44) + 1);
            v22 = *(_DWORD *)(v13 + 44) + 1;
          }
          else
          {
            v21 = 0;
            v22 = 0;
          }
          if ( v22 >= v19 )
            BUG();
          v23 = *(__int64 **)(*(_QWORD *)(v20 + 8 * v21) + 8LL);
          if ( !v23 )
            break;
          v13 = *v23;
          if ( *v23 == v12 || *v23 == 0 || !v12 )
            break;
          sub_B196A0(a6, v12, v13);
          if ( v24 )
          {
            v25 = *(_BYTE **)(v39 + 24);
            if ( *v25 <= 0x1Cu )
              v25 = 0;
            if ( (unsigned __int8)sub_2C9EA10(a1, (unsigned __int64)v25, v12, a6) )
            {
              v37 = (_BYTE *)v9[3];
              if ( *v37 <= 0x1Cu )
                v37 = 0;
              v15 = sub_2C9EA10(a1, (unsigned __int64)v37, v13, a6);
              if ( (_BYTE)v15 )
              {
                *a4 = v12;
                *a5 = v13;
                return v15;
              }
            }
            return 0;
          }
        }
      }
      return 0;
    }
    *a4 = v12;
    *a5 = v13;
    if ( a7 )
    {
      *a7 = 0;
      return v15;
    }
    return 1;
  }
  v28 = a2[2];
  if ( v28 > 1 )
    v10 = *(_QWORD *)(v8[v28 - 1] + 16);
  v15 = sub_2C9D230(a1, a6, v10, v11);
  if ( (_BYTE)v15 )
  {
    *a4 = *(_QWORD *)(**(_QWORD **)a2 + 16LL);
    *a5 = v9[2];
    if ( a7 )
    {
      *a7 = 1;
      return v15;
    }
    return 1;
  }
  return 0;
}
