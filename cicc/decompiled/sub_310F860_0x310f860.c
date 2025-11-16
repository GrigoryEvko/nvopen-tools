// Function: sub_310F860
// Address: 0x310f860
//
__int64 __fastcall sub_310F860(_BYTE *a1)
{
  if ( sub_B2FC80((__int64)a1) || (unsigned __int8)sub_B2FC00(a1) )
    return 0;
  else
    return (unsigned int)sub_B2D610((__int64)a1, 20) ^ 1;
}
