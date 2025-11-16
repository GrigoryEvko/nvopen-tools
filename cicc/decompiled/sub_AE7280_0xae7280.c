// Function: sub_AE7280
// Address: 0xae7280
//
__int64 __fastcall sub_AE7280(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4)
{
  int v6; // eax
  unsigned __int8 **v8; // rax
  unsigned __int8 **v9; // rdx
  unsigned __int8 **v10; // rax
  unsigned __int8 **v11; // rdx
  unsigned __int8 **v12; // rax
  __int64 v13; // rcx
  unsigned __int8 **v14; // rdx
  char v15; // dl
  unsigned __int8 **v16; // rax
  __int64 v17; // rdx
  unsigned __int8 **v18; // r15
  unsigned __int8 **v19; // rax
  __int64 v20; // rcx
  unsigned __int8 **v21; // rdx
  unsigned __int8 **v22; // [rsp-40h] [rbp-40h]

  if ( a4 )
  {
    v6 = *a4;
    if ( (unsigned __int8)(v6 - 5) > 0x1Fu )
      return 0;
    if ( (_BYTE)v6 == 6 )
      return 1;
    if ( *(_BYTE *)(a2 + 28) )
    {
      v8 = *(unsigned __int8 ***)(a2 + 8);
      v9 = &v8[*(unsigned int *)(a2 + 20)];
      if ( v8 != v9 )
      {
        while ( a4 != *v8 )
        {
          if ( v9 == ++v8 )
            goto LABEL_12;
        }
        return 1;
      }
    }
    else if ( sub_C8CA60(a2, a4, (unsigned int)(v6 - 5), a4) )
    {
      return 1;
    }
LABEL_12:
    if ( *(_BYTE *)(a3 + 28) )
    {
      v10 = *(unsigned __int8 ***)(a3 + 8);
      v11 = &v10[*(unsigned int *)(a3 + 20)];
      if ( v10 == v11 )
        return 0;
      while ( a4 != *v10 )
      {
        if ( v11 == ++v10 )
          return 0;
      }
    }
    else if ( !sub_C8CA60(a3, a4, v9, a4) )
    {
      return 0;
    }
    if ( !*(_BYTE *)(a1 + 28) )
      goto LABEL_24;
    v12 = *(unsigned __int8 ***)(a1 + 8);
    v13 = *(unsigned int *)(a1 + 20);
    v14 = &v12[v13];
    if ( v12 != v14 )
    {
      while ( a4 != *v12 )
      {
        if ( v14 == ++v12 )
          goto LABEL_37;
      }
      return 0;
    }
LABEL_37:
    if ( (unsigned int)v13 < *(_DWORD *)(a1 + 16) )
    {
      *(_DWORD *)(a1 + 20) = v13 + 1;
      *v14 = a4;
      ++*(_QWORD *)a1;
    }
    else
    {
LABEL_24:
      sub_C8CC70(a1, a4);
      if ( !v15 )
        return 0;
    }
    v16 = (unsigned __int8 **)sub_A17150(a4 - 16);
    v22 = &v16[v17];
    v18 = v16;
    if ( v16 != v22 )
    {
      while ( a4 == *v18 || (unsigned __int8)sub_AE7280(a1, a2, a3) )
      {
        if ( v22 == ++v18 )
          goto LABEL_29;
      }
      return 0;
    }
LABEL_29:
    if ( !*(_BYTE *)(a2 + 28) )
      goto LABEL_39;
    v19 = *(unsigned __int8 ***)(a2 + 8);
    v20 = *(unsigned int *)(a2 + 20);
    v21 = &v19[v20];
    if ( v19 != v21 )
    {
      while ( a4 != *v19 )
      {
        if ( v21 == ++v19 )
          goto LABEL_40;
      }
      return 1;
    }
LABEL_40:
    if ( (unsigned int)v20 < *(_DWORD *)(a2 + 16) )
    {
      *(_DWORD *)(a2 + 20) = v20 + 1;
      *v21 = a4;
      ++*(_QWORD *)a2;
    }
    else
    {
LABEL_39:
      sub_C8CC70(a2, a4);
    }
    return 1;
  }
  return 0;
}
