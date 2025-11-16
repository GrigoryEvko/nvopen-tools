// Function: sub_E5EDD0
// Address: 0xe5edd0
//
__int64 __fastcall sub_E5EDD0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  unsigned int v4; // edx
  char v5; // r15
  char *v6; // rax
  char v7; // si
  unsigned int v8; // ecx
  char v9; // al
  __int64 v10; // r14
  char v11; // bl
  char *v12; // rax
  char *v14; // rax
  __int64 v15; // [rsp+10h] [rbp-90h]
  unsigned int v16; // [rsp+1Ch] [rbp-84h]
  unsigned int v17; // [rsp+1Ch] [rbp-84h]
  __int64 v18; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v19[3]; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v20; // [rsp+48h] [rbp-58h]
  char *v21; // [rsp+50h] [rbp-50h]
  __int64 v22; // [rsp+58h] [rbp-48h]
  __int64 v23; // [rsp+60h] [rbp-40h]

  v2 = a2;
  v15 = *(_QWORD *)(a2 + 48);
  sub_E81940(*(_QWORD *)(a2 + 112), &v18, a1);
  *(_QWORD *)(a2 + 48) = 0;
  v22 = 0x100000000LL;
  v19[1] = 2;
  v19[2] = 0;
  v19[0] = &unk_49DD288;
  v20 = 0;
  v21 = 0;
  v23 = a2 + 40;
  sub_CB5980((__int64)v19, 0, 0, 0);
  v3 = v18;
  v4 = 1;
  *(_DWORD *)(a2 + 80) = 0;
  while ( 1 )
  {
    v9 = v3;
    v7 = v3 & 0x7F;
    v3 >>= 7;
    if ( v3 )
    {
      if ( v3 != -1 || (v9 & 0x40) == 0 )
      {
        v5 = 1;
        goto LABEL_3;
      }
    }
    else
    {
      v5 = 1;
      if ( (v9 & 0x40) != 0 )
        goto LABEL_3;
    }
    v5 = 0;
    if ( (unsigned int)v15 <= v4 )
      break;
LABEL_3:
    v6 = v21;
    v7 |= 0x80u;
    if ( (unsigned __int64)v21 >= v20 )
      goto LABEL_12;
LABEL_4:
    v21 = v6 + 1;
    v8 = v4 + 1;
    *v6 = v7;
    if ( !v5 )
      goto LABEL_13;
LABEL_5:
    v4 = v8;
  }
  v6 = v21;
  if ( (unsigned __int64)v21 < v20 )
    goto LABEL_4;
LABEL_12:
  v16 = v4;
  sub_CB5D20((__int64)v19, v7);
  v4 = v16;
  v8 = v16 + 1;
  if ( v5 )
    goto LABEL_5;
LABEL_13:
  if ( (unsigned int)v15 > v4 )
  {
    v10 = v3 >> 63;
    v11 = (v3 >> 63) | 0x80;
    if ( v4 < (int)v15 - 1 )
    {
      while ( 1 )
      {
        v14 = v21;
        if ( (unsigned __int64)v21 < v20 )
        {
          ++v21;
          *v14 = v11;
        }
        else
        {
          v17 = v8;
          sub_CB5D20((__int64)v19, v11);
          v8 = v17;
        }
        if ( v8 == (_DWORD)v15 - 1 )
          break;
        ++v8;
      }
    }
    v12 = v21;
    if ( (unsigned __int64)v21 >= v20 )
    {
      sub_CB5D20((__int64)v19, v10 & 0x7F);
    }
    else
    {
      ++v21;
      *v12 = v10 & 0x7F;
    }
  }
  LOBYTE(v2) = *(_QWORD *)(v2 + 48) != v15;
  v19[0] = &unk_49DD388;
  sub_CB5840((__int64)v19);
  return (unsigned int)v2;
}
