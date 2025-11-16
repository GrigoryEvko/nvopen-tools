// Function: sub_5E6DB0
// Address: 0x5e6db0
//
_QWORD *__fastcall sub_5E6DB0(__int64 a1, __int64 a2, int a3, _DWORD *a4, _QWORD *a5)
{
  __int64 v7; // rax
  _QWORD *result; // rax
  _QWORD *v9; // r12
  __int64 v10; // rax
  __int64 v11; // r14
  _QWORD *v12; // r13
  __int64 v13; // r12
  __int64 v14; // rdx
  char v15; // si
  char v16; // di
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  char v21; // dl
  __int64 v22; // rax
  char v23; // dl
  __int64 v24; // rdi
  __int64 v25; // rax
  char v26; // al
  __int64 v27; // rdx
  __int64 v28; // rax
  unsigned __int8 v29; // al
  char v30; // al
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  _DWORD *i; // [rsp+0h] [rbp-60h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  unsigned int v36; // [rsp+14h] [rbp-4Ch]
  __int64 v38; // [rsp+20h] [rbp-40h]
  _QWORD *v39; // [rsp+28h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 40);
  for ( i = a4; *(_BYTE *)(v7 + 140) == 12; v7 = *(_QWORD *)(v7 + 160) )
    ;
  result = *(_QWORD **)(*(_QWORD *)v7 + 96LL);
  v9 = (_QWORD *)result[6];
  if ( !a3 )
    v9 = (_QWORD *)result[5];
  v39 = v9;
  if ( v9 )
  {
    v10 = a1;
    if ( *(_BYTE *)(a1 + 140) == 12 )
    {
      do
        v10 = *(_QWORD *)(v10 + 160);
      while ( *(_BYTE *)(v10 + 140) == 12 );
    }
    else
    {
      v10 = a1;
    }
    v38 = *(_QWORD *)(*(_QWORD *)v10 + 96LL);
    while ( 1 )
    {
LABEL_9:
      v11 = v39[1];
      v12 = *(_QWORD **)(v38 + 48);
      if ( !a3 )
        v12 = *(_QWORD **)(v38 + 40);
      if ( v12 )
        break;
LABEL_45:
      v26 = *(_BYTE *)(v11 + 80);
      v27 = v11;
      if ( v26 == 16 )
      {
        v27 = **(_QWORD **)(v11 + 88);
        v26 = *(_BYTE *)(v27 + 80);
      }
      if ( v26 == 24 )
        v27 = *(_QWORD *)(v27 + 88);
      v28 = sub_8793F0(*(_QWORD *)(v27 + 64), a1, a2);
      v35 = sub_87F190(v11, a1, v28, 0, 0);
      v36 = *(unsigned __int8 *)(*(_QWORD *)(a2 + 112) + 25LL);
      v29 = sub_87D550(v11);
      v30 = sub_87D600(v29, v36) & 3 | *(_BYTE *)(v35 + 96) & 0xFC;
      *(_BYTE *)(v35 + 96) = v30;
      if ( *(_BYTE *)(v11 + 80) == 16 )
        *(_BYTE *)(v35 + 96) = *(_BYTE *)(v11 + 96) & 8 | v30 & 0xF7;
      sub_5E6C40(v35, v38);
      *i = 1;
      result = (_QWORD *)*v39;
      v39 = result;
      if ( !result )
        return result;
    }
    while ( 1 )
    {
      v13 = v12[1];
      if ( *(_QWORD *)v13 == *(_QWORD *)v11 )
        goto LABEL_17;
      if ( !a3 )
        goto LABEL_14;
      v21 = *(_BYTE *)(v13 + 80);
      v22 = v12[1];
      if ( v21 == 16 )
      {
        v22 = **(_QWORD **)(v13 + 88);
        v21 = *(_BYTE *)(v22 + 80);
      }
      if ( v21 == 24 )
        v22 = *(_QWORD *)(v22 + 88);
      v23 = *(_BYTE *)(v11 + 80);
      v24 = *(_QWORD *)(*(_QWORD *)(v22 + 88) + 176LL);
      v25 = v11;
      if ( v23 == 16 )
      {
        v25 = **(_QWORD **)(v11 + 88);
        v23 = *(_BYTE *)(v25 + 80);
      }
      if ( v23 == 24 )
        v25 = *(_QWORD *)(v25 + 88);
      if ( (unsigned int)sub_8D97D0(
                           *(_QWORD *)(*(_QWORD *)(v24 + 152) + 160LL),
                           *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v25 + 88) + 176LL) + 152LL) + 160LL),
                           8,
                           a4,
                           a5) )
      {
LABEL_17:
        if ( *(_BYTE *)(v13 + 80) != 16 || (*(_BYTE *)(v13 + 96) & 4) != 0 )
          goto LABEL_33;
        a5 = *(_QWORD **)(v13 + 88);
        v14 = *a5;
        v15 = *(_BYTE *)(*a5 + 80LL);
        if ( v15 == 24 )
        {
          v14 = *(_QWORD *)(v14 + 88);
          v15 = *(_BYTE *)(v14 + 80);
        }
        v16 = *(_BYTE *)(v11 + 80);
        v17 = v11;
        if ( v16 == 16 )
        {
          v17 = **(_QWORD **)(v11 + 88);
          v16 = *(_BYTE *)(v17 + 80);
        }
        if ( v16 == 24 )
          v17 = *(_QWORD *)(v17 + 88);
        v18 = *(_QWORD *)(v14 + 88);
        v19 = *(_QWORD *)(v17 + 88);
        if ( v15 == 20 )
        {
          v31 = *(_QWORD *)(*(_QWORD *)(v18 + 104) + 200LL);
          v32 = *(_QWORD *)(*(_QWORD *)(v19 + 104) + 200LL);
          if ( v31 == v32 )
            goto LABEL_33;
          if ( v31 )
          {
            if ( v32 )
            {
              if ( dword_4F07588 )
              {
                v33 = *(_QWORD *)(v31 + 32);
                if ( *(_QWORD *)(v32 + 32) == v33 )
                {
                  if ( v33 )
                    goto LABEL_33;
                }
              }
            }
          }
        }
        else if ( v18 == v19
               || v18
               && v19
               && (a4 = (_DWORD *)dword_4F07588, dword_4F07588)
               && (v20 = *(_QWORD *)(v18 + 32), *(_QWORD *)(v19 + 32) == v20)
               && v20
               || (*(_BYTE *)(a2 + 96) & 2) != 0 && (unsigned int)sub_8D5D50(a2, a5[1]) )
        {
LABEL_33:
          result = (_QWORD *)*v39;
          v39 = result;
          if ( !result )
            return result;
          goto LABEL_9;
        }
LABEL_14:
        v12 = (_QWORD *)*v12;
        if ( !v12 )
          goto LABEL_45;
      }
      else
      {
        v12 = (_QWORD *)*v12;
        if ( !v12 )
          goto LABEL_45;
      }
    }
  }
  return result;
}
