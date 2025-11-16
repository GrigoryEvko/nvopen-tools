// Function: sub_8CF3D0
// Address: 0x8cf3d0
//
__int64 __fastcall sub_8CF3D0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // r13
  int v3; // ebx
  __int64 v4; // r14
  __int64 *v5; // rax
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rbx
  __int64 v10; // r14
  __int64 v11; // rbx
  __int64 i; // r12

  result = *(_QWORD *)(a1 + 32);
  if ( !result || qword_4F074B0 != qword_4F60258 )
    return result;
  v2 = *(_QWORD *)result;
  if ( a1 != *(_QWORD *)result )
  {
    result = (unsigned int)*(unsigned __int8 *)(v2 + 140) - 9;
    if ( (unsigned __int8)(*(_BYTE *)(v2 + 140) - 9) > 2u )
      return result;
    v3 = sub_8D23B0(v2);
    if ( v3 )
    {
      **(_QWORD **)(a1 + 32) = a1;
      if ( (unsigned int)sub_8D2490(a1) )
        sub_8C9FB0(a1, 1u);
    }
    else
    {
      if ( (*(_BYTE *)(a1 - 8) & 2) != 0 )
      {
        v4 = v2;
        v2 = a1;
        goto LABEL_10;
      }
      **(_QWORD **)(a1 + 32) = a1;
    }
    v5 = *(__int64 **)(v2 + 32);
    if ( v5 )
      v4 = *v5;
    else
      v4 = v2;
    v3 = 1;
LABEL_10:
    sub_8CAE10(v2);
    sub_8CCD60();
    result = sub_8D23B0(v4);
    if ( !(_DWORD)result )
    {
      result = sub_8D23B0(v2);
      if ( !(_DWORD)result )
      {
        if ( v3 )
        {
          result = sub_8CE860(v2);
          if ( (_DWORD)result )
          {
            result = *(_QWORD *)(v2 + 168);
            v6 = *(_QWORD *)(result + 152);
            if ( v6 )
            {
              v7 = *(_QWORD *)(v2 + 32);
              v8 = v6;
              if ( v7 )
                v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v7 + 168LL) + 152LL);
              v9 = sub_8C6270(*(_QWORD *)(v8 + 144));
              result = sub_8C6270(*(_QWORD *)(v6 + 144));
              v10 = result;
              if ( result && v9 )
              {
                do
                {
                  if ( (*(_BYTE *)(v9 + 193) & 0x10) == 0 )
                    sub_899FE0(v9, v10);
                  v9 = sub_8C6270(*(_QWORD *)(v9 + 112));
                  result = sub_8C6270(*(_QWORD *)(v10 + 112));
                  v10 = result;
                }
                while ( v9 && result );
              }
              v11 = *(_QWORD *)(v8 + 112);
              for ( i = *(_QWORD *)(v6 + 112); v11; i = *(_QWORD *)(i + 112) )
              {
                if ( !i )
                  break;
                result = sub_89A000(v11, i);
                v11 = *(_QWORD *)(v11 + 112);
              }
            }
          }
        }
        else
        {
          result = dword_4D03FC0;
          if ( dword_4D03FC0 )
            return sub_8CE860(v2);
        }
      }
    }
    return result;
  }
  result = sub_8D2490(a1);
  if ( (_DWORD)result )
    return (__int64)sub_8C9FB0(a1, 1u);
  return result;
}
